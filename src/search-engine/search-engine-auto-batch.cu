#include <search-engine/search-engine-auto-batch.cuh>

static bool write_coeff(const std::string& file_path, const double coeff[4]) {
    std::ofstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << file_path << std::endl;
        return false;
    }

    file.write(reinterpret_cast<const char*>(coeff), sizeof(double) * 4);
    if (!file) {
        std::cerr << "Failed to write data to file: " << file_path << std::endl;
        return false;
    }
    file.close();
    return true;
}

#define MAX_TMP_RES_BUFF_SIZE 256 * 1024 * 1024
static void flush_current_res(float *dis, uint64_t* idx, size_t size, cudaEvent_t &dis_copy_done_event, 
                       cudaEvent_t &idx_copy_done_event, const cudaEvent_t &compute_done_event, int device_id, std::string const& path = "res/",
                       bool force_flush = false, bool overwrite = false, bool reset_offset = false) 
{
    thread_local float* dis_buff = nullptr;
    thread_local uint64_t* indices_buff = nullptr;
    thread_local size_t offset = 0;
    thread_local size_t buff_size = MAX_TMP_RES_BUFF_SIZE;
    thread_local size_t file_offset = 0;
    thread_local int thread_id = -1;
    thread_local cudaStream_t dis_copy_stream = nullptr;
    thread_local cudaStream_t idx_copy_stream = nullptr;

    if (reset_offset) {
      offset = 0;
      file_offset = 0;
    }

    if (thread_id == -1) {
        // Use a stable hashing method to ensure consistent thread ID
        thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 10000;
    }

    // Generate file paths based on device_id to ensure dictionary order and consistency
    std::string dis_file_path = path + "distances_" + std::to_string(device_id);
    std::string neigh_file_path = path + "neighbors_" + std::to_string(device_id);

    // Initialize CUDA streams if not already created
    if (dis_copy_stream == nullptr) {
        cudaStreamCreate(&dis_copy_stream);
    }
    if (idx_copy_stream == nullptr) {
        cudaStreamCreate(&idx_copy_stream);
    }

    // Allocate buffers if not already allocated
    if (dis_buff == nullptr) {
        dis_buff = new float[buff_size];
    }
    if (indices_buff == nullptr) {
        indices_buff = new uint64_t[buff_size];
    }

    // If buffer is full or forced to flush, write to disk
    if (offset + size >= buff_size || force_flush) {

        LOG(INFO) << "device: " << device_id << ", flush tmp res with offset: " << 
                offset << "with file offset: " << file_offset << ", overwrite: " << overwrite;
        auto dis_write_future = 
            write_binary_file_async(dis_file_path, file_offset * sizeof(float), (void*)dis_buff, offset * sizeof(float), !overwrite);
        auto neigh_write_future = 
            write_binary_file_async(neigh_file_path, file_offset * sizeof(uint64_t), (void*)indices_buff, offset * sizeof(uint64_t), !overwrite);

        if (force_flush) {
            // Wait for async writes to complete
            bool res = dis_write_future.get() && neigh_write_future.get();
            if (!res) {
                throw std::runtime_error("Failed to flush query result!");
            }
        }
        file_offset += offset;
        offset = 0; // Reset offset after flush
    }

    // If there is new data to process, copy it to the buffers asynchronously
    if (dis != nullptr && idx != nullptr) {
        assert(dis_copy_done_event != nullptr);
        assert(idx_copy_done_event != nullptr);
        assert(compute_done_event != nullptr);

        cudaStreamWaitEvent(dis_copy_stream, compute_done_event, 0);
        checkCUDAErrorWithLine("Insert wait for compute stream failed!");
        cudaMemcpyAsync(dis_buff + offset, dis, sizeof(float) * size, cudaMemcpyDeviceToHost, dis_copy_stream);
        checkCUDAErrorWithLine("Copy temporary buffer to host failed!");
        cudaEventRecord(dis_copy_done_event, dis_copy_stream);

        cudaStreamWaitEvent(idx_copy_stream, compute_done_event, 0);
        cudaMemcpyAsync(indices_buff + offset, idx, sizeof(uint64_t) * size, cudaMemcpyDeviceToHost, idx_copy_stream);
        checkCUDAErrorWithLine("Copy temporary buffer to host failed!");
        cudaEventRecord(idx_copy_done_event, idx_copy_stream);
    
        offset += size;
    }
}

static bool read_coeff(const std::string& file_path, double coeff[4]) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for reading: " << file_path << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(coeff), sizeof(double) * 4);
    if (!file) {
        std::cerr << "Failed to read data from file: " << file_path << std::endl;
        return false;
    }
    file.close();
    return true;
}

auto_batch_search::auto_batch_search(
    std::string const& algorithm, 
    std::string const& config_path,
    std::string const& filter_config_path 
) : f_config(filter_config_path) 
{
    config = config_factory::create(algorithm, config_path);
    index = index_factory::create(algorithm, config.get(), filter_config_path);
    parafilter_mmr::init_mmr();
    
    std::string dataset_path = config->path;
    get_data_set_list(keys, std::string(dataset_path + "keys").c_str());
    get_data_type_list(types, std::string(dataset_path + "dtypes").c_str(), keys);

    el::Configurations conf("myeasylogger.conf");
    el::Loggers::reconfigureLogger("default", conf);
    ::build_dataset(keys, types, nullptr, size_map, dataset_path);

    // initilize task meta
    n_data = size_map["train_vec"].first;
    n_dim     = size_map["train_vec"].second;
    n_queries = size_map["test_vec"].first;
    l = f_config.l;
    label_dim = size_map["train_label"].second / l;
    filter_dim = size_map["test_label"].second / l;

    query_time = 0;
    build_time = 0;
}

void auto_batch_search::run() 
{    
    double coeff[4];
    if (config->is_calc_mem_predictor_coeff) {
        calc_mem_coeff(coeff);
        write_coeff("coeff", coeff);
    }
    else read_coeff("coeff", coeff);

    uint64_t data_batch_size;
    uint64_t query_batch_size;

    split_task(coeff, query_batch_size, data_batch_size);

    int device_count;
    cudaGetDeviceCount(&device_count);
    if (!config->enable_multi_gpu) device_count = 1;
    bool break_down = config->break_down;

    auto per_device_worker = [&](uint32_t i) {
        cudaSetDevice(i);
        Timer global_timer;

        // fixme: avoid to use raft mem pool
        raft::device_resources dev_resources;
        rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
        rmm::mr::get_current_device_resource(), 1 * 1024 * 1024 * 1024ull);
        rmm::mr::set_current_device_resource(&pool_mr);

        uint64_t query_offset, n_queries_device;
        uniformly_divide(n_queries, device_count, i, n_queries_device, query_offset);
        
        LOG(TRACE) << "device :" << i << " choose data batach size:" 
            << data_batch_size << " query batch size" <<  query_batch_size;

        uint64_t inter_buffer_size = config->topk * query_batch_size;
        // todo: ping pang the output buffer to increase throughput
        float* selected_distance_device_ptr;
        cudaMalloc((void **)&selected_distance_device_ptr, SWAP_BUFF_COUNT * inter_buffer_size * sizeof(float));
        uint64_t* selected_indices_device_ptr;
        cudaMalloc((void **)&selected_indices_device_ptr, SWAP_BUFF_COUNT * inter_buffer_size * sizeof(uint64_t));
        
        std::vector<cudaEvent_t> dis_copy_done_event(SWAP_BUFF_COUNT);
        std::vector<cudaEvent_t> idx_copy_done_event(SWAP_BUFF_COUNT);
        std::vector<cudaEvent_t> compute_done_event(SWAP_BUFF_COUNT);

        for (int buff_id = 0; buff_id < SWAP_BUFF_COUNT; buff_id++) {
            cudaEventCreate(&dis_copy_done_event[buff_id]);
            cudaEventCreate(&idx_copy_done_event[buff_id]);
            cudaEventCreate(&compute_done_event[buff_id]);
            checkCUDAErrorWithLine("failed to create flush event");
        }
        std::map<std::string, void*> data_map;
        int cur_res_buff_offset = 0;
        uint64_t topk = config->topk;

        build_input_t build_in;
        query_input_t query_in;
        query_output_t query_output;
        for (uint64_t data_batch_offset = 0; data_batch_offset < n_data; data_batch_offset += data_batch_size) {
            uint64_t cur_data_batch_size;
            cur_data_batch_size = data_batch_size;

            for (uint64_t query_batch_offset = 0; query_batch_offset < n_queries_device; query_batch_offset += query_batch_size) {        
                uint64_t cur_query_batch_size = query_batch_size;
                uint64_t cur_query_offset = query_offset + query_batch_offset;

                build_dataset(config->path, data_batch_offset, cur_data_batch_size, cur_query_offset, cur_query_batch_size, build_in, query_in);

                query_output.distances = raft::make_device_matrix_view<float, uint64_t>(selected_distance_device_ptr + cur_res_buff_offset * inter_buffer_size, 
                        cur_query_batch_size, topk);
                query_output.neighbors = raft::make_device_matrix_view<uint64_t, uint64_t>(selected_indices_device_ptr + cur_res_buff_offset * inter_buffer_size, 
                        cur_query_batch_size, topk);
                
                if (query_batch_offset == 0) { 
                    parafilterPerfLogWraper(index->build(build_in), build_time);
                }
                parafilterPerfLogWraper(index->query(query_in, query_output), query_time);

                cudaEventSynchronize(dis_copy_done_event[cur_res_buff_offset]);
                cudaEventSynchronize(idx_copy_done_event[cur_res_buff_offset]);

                cudaEventRecord(compute_done_event[cur_res_buff_offset]);  
                flush_current_res(query_output.distances.data_handle(), query_output.neighbors.data_handle(), inter_buffer_size, 
                            dis_copy_done_event[cur_res_buff_offset], idx_copy_done_event[cur_res_buff_offset], 
                            compute_done_event[cur_res_buff_offset], i);
                
                parafilter_mmr::free_cur_workspace_device_mems(false);
                cur_res_buff_offset = (cur_res_buff_offset + 1) % SWAP_BUFF_COUNT;
                cudaDeviceSynchronize();
            }
        }
        cudaDeviceSynchronize();
        LOG(TRACE) << "device: " << i << "build time:" << build_time << ", query time:" << query_time;
        parafilter_mmr::free_cur_workspace_device_mems();
        cudaFree(selected_distance_device_ptr);
        cudaFree(selected_indices_device_ptr);
        flush_current_res((float*)0, (uint64_t*)0, 0, dis_copy_done_event[0], idx_copy_done_event[0], compute_done_event[0], 
                        i, "res/", true);
        if (data_batch_size != n_data) {
            uint64_t batch_size = (n_data + data_batch_size - 1) / data_batch_size;
            auto merged_dis_view = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries_device, topk);
            auto merged_idx_view = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries_device, topk);

            parafilterPerfLogWraper(
                (merge_intermediate_result)(
                                        dev_resources,
                                        "res/", 
                                        batch_size, 
                                        data_batch_size, 
                                        n_queries_device, 
                                        topk, 
                                        0l, 
                                        i, 
                                        merged_dis_view, 
                                        merged_idx_view), 
                query_time
            );
            cudaEventRecord(compute_done_event[0]);
            // todo: process the case when output buffer large than the tmp buffer
            flush_current_res(merged_dis_view.data_handle(), merged_idx_view.data_handle(), 
                                topk * n_queries_device, dis_copy_done_event[0], idx_copy_done_event[0], compute_done_event[0], 
                                i, "res/", false, false, true);

            cudaEventSynchronize(dis_copy_done_event[0]);
            cudaEventSynchronize(idx_copy_done_event[0]);

            flush_current_res((float*)0, (uint64_t*)0, 0, dis_copy_done_event[0], idx_copy_done_event[0], compute_done_event[0], 
                            i, "res/", true, true);
            parafilter_mmr::free_cur_workspace_device_mems();
        }  
    };

    std::vector<std::thread> workers;
    for (uint32_t device_id = 0; device_id < device_count; device_id++) {
        workers.emplace_back(per_device_worker, device_id);
    }

    for(auto &w: workers) w.join();

    float recall = compute_recall("./res", std::string(config->path + "neighbors"), config->topk, n_queries);
    LOG(TRACE) << "final recall:" << recall;
}
void auto_batch_search::build_dataset(
    std::string dataset_path, 
    uint64_t data_offset, 
    uint64_t data_size, 
    uint64_t query_offset, 
    uint64_t query_size, 
    build_input_t &build_in,
    query_input_t &query_in
) 
{
    thread_local std::map<std::string, void*> data_map;
    ::build_dataset(keys, types, &data_map, size_map, dataset_path, data_offset, data_size, query_offset, query_size, filter_dim);

    build_in.dataset = raft::make_device_matrix_view<float, uint64_t>((float *)data_map["train_vec"], data_size, n_dim);
    query_in.queries = raft::make_device_matrix_view<float, uint64_t>((float *)data_map["test_vec"], query_size, n_dim);
    query_in.dataset = build_in.dataset;

    build_in.data_labels = raft::make_device_matrix_view<float, uint64_t>((float *)data_map["train_label"], data_size, l);
    query_in.query_labels = raft::make_device_matrix_view<float, uint64_t>((float *)data_map["test_label"], query_size, l * filter_dim);
}
void auto_batch_search::calc_mem_coeff(double *coeff) 
{
    parafilter_mmr::reset_current_workspace(2ull * 1024 * 1024 * 1024);
    raft::device_resources dev_resources;
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1 * 1024 * 1024 * 1024ull);
    rmm::mr::set_current_device_resource(&pool_mr);

    uint64_t data_batches_size[4] = {
        10000, 20000, 30000, 40000
    };
    uint64_t query_batches_size[4] = {
        5, 10, 25, 20
    };

    build_input_t build_in;
    query_input_t query_in;

    double mat[4][5];
    auto write_mat = [&mat] (int row, size_t fake_data, size_t fake_query, uint64_t mem_used) {
        mat[row][0] = fake_data;
        mat[row][1] = fake_query;
        mat[row][2] = fake_data * fake_query;
        mat[row][3] = 1;
        mat[row][4] = mem_used;
    };

    for (int i = 0; i < 4; i++) {
        query_output_t out;

        out.neighbors = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(query_batches_size[i], config->topk);
        out.distances = parafilter_mmr::make_device_matrix_view<float, uint64_t>(query_batches_size[i], config->topk);

        build_dataset(config->path, 0, data_batches_size[i], 0, query_batches_size[i], build_in, query_in);
        
        index->build(build_in);
        index->query(query_in, out);

        uint64_t used = parafilter_mmr::get_current_workspace_used();
        write_mat(i, data_batches_size[i], query_batches_size[i], used);
        parafilter_mmr::free_cur_workspace_device_mems();
        parafilter_mmr::reset_current_workspace(2ull * 1024 * 1024 * 1024);
    }

    gauss(mat, 4);
    for (int i = 0; i < 4; i++)
        coeff[i] = mat[i][4];
}

void auto_batch_search::split_task(
                double* coeff,
                uint64_t &query_batch_size, 
                uint64_t &data_batch_size, 
                uint64_t aditional, 
                uint64_t lowest_query_batch_size)  
{
    uint64_t topk = config->topk;
    uint64_t mem_bound = config->mem_bound;

    uint64_t available, total;
    get_current_device_mem_info(available, total);
    int id;
    cudaGetDevice(&id);
    LOG(INFO) << "device: " << id << "available: " << available << " upper bound: "
                << mem_bound;

    uint64_t upper_bound = std::min(available / 5, mem_bound);

    auto bisearch_proper_split = [coeff, topk, this,  &query_batch_size, &data_batch_size] (uint64_t upper_bound) {
        uint64_t r_d = n_data;
        uint64_t l_d = topk;

        uint64_t r_q = n_queries;
        uint64_t l_q = 1;

        while (1) {
            uint64_t mid_d = (r_d + l_d + 1) >> 1;

            r_q = n_queries;
            l_q = 1;

            while (l_q < r_q) {
                uint64_t mid_q = (l_q + r_q + 1) >> 1;
                uint64_t value = coeff[0] * mid_d + 
                                coeff[1] * mid_q +
                                coeff[2] * mid_d * mid_q + 
                                coeff[3] + 1;

                if (value > upper_bound) r_q = mid_q - 1;
                else l_q = mid_q; 
            } 
            if (l_q < 125) r_d = mid_d - 1;
            else l_d = mid_d;
            if (l_d == r_d && l_q >= 125) break;
        }

        query_batch_size = l_q;
        data_batch_size = l_d;
    };
    // todo: when cannot find proper batch size for current upper bound, enlarge it
    bisearch_proper_split(upper_bound);
    while (n_queries % query_batch_size != 0) query_batch_size--;
    data_batch_size = findMaxFactor(data_batch_size, n_data);
}