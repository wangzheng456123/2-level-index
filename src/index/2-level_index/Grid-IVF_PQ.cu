#include <index/2-level_index/Grid-IVF_PQ.cuh>
namespace Grid_IVF_PQ {
    __device__ bool hyperrects_intersect(const float* rect1, const float* rect2, int l) 
    {
        for (int i = 0; i < l; ++i) {
            float l1 = rect1[2 * i];
            float r1 = rect1[2 * i + 1];
            float l2 = rect2[2 * i];
            float r2 = rect2[2 * i + 1];

            if (r1 < l2 || r2 < l1) {
                return false;
            }
        }
            return true;
    }

    __global__ void intersect_kernel(
        const float* queries,    // n_queries * 2 * l
        const float* grids,      // grids_cnt * 2 * l
        int* bitmap_matrix,      // n_queries * grids_cnt
        int n_queries,
        int grids_cnt,
        int l
    ) 
    {
        int q_idx = blockIdx.y * blockDim.y + threadIdx.y; 
        int g_idx = blockIdx.x * blockDim.x + threadIdx.x; 

        if (q_idx >= n_queries || g_idx >= grids_cnt) return;

        const float* query_ptr = queries + q_idx * 2 * l;
        const float* grid_ptr  = grids + g_idx * 2 * l;

        int result = hyperrects_intersect(query_ptr, grid_ptr, l) ? 1 : 0;
        bitmap_matrix[q_idx * grids_cnt + g_idx] = result;
    }

    uint64_t index::estimate_filter_ratio(
        query_input_t const& query_in    
    ) const
    {
        if (estimated_filter_ratio) return estimated_filter_ratio;
        else {
            assert(false && "no way to estimate filter ratio when query currently");
        }
        return 0;
    }

    void index::generate_grid_meta(
        build_input_t const& in, 
        std::vector<grid_meta> &grid_info_host
    )
    {
        uint64_t l = f_config.l;
        std::vector<float> global_min;
        std::vector<float> global_max;
        global_min.resize(l);
        global_max.resize(l);
        double max_grids_cnt = static_cast<double>(grids_config.max_grids_cnt);

        calculate_batch_min_max(in.data_labels, global_min, global_max, l);
        estimated_filter_ratio = 1.f;
        for (uint64_t i = 0; i < l; i++) {
            grid_info_host[i].start = global_min[i];
            grid_info_host[i].end = global_max[i];

            int type = f_config.filter_type[i];
            float ranges_len = grid_info_host[i].end - grid_info_host[i].start;
            if (type == 0) 
                ranges_len = f_config.shift_val[2 * i + 1] + f_config.shift_val[2 * i];
            else if (type == 1) {
                std::vector<int> &cur_map = f_config.interval_map[i];
                for (int j = 0; j < cur_map.size() / 2; j++) {
                    ranges_len = 0;
                    ranges_len = max(ranges_len, abs(float(cur_map[2 * j + 1] - cur_map[2 * j])));
                }
            }
            else {
                assert(false && "Unimplement filter type");
            }

            ranges_len *= 1.1;
            int cnt = (global_max[i] - global_min[i]) / ranges_len + 1;
            if (cnt >= max_grids_cnt) {
                uint64_t overlape = max_grids_cnt >= 2 ? 2 : 1;
                estimated_filter_ratio *= (grid_info_host[i].end - grid_info_host[i].start) * overlape / 
                    (ranges_len * int(max_grids_cnt));
                grid_info_host[i].cnt = max_grids_cnt;
                grid_info_host[i].len = (global_max[i] - global_min[i]) / max_grids_cnt;
                max_grids_cnt = 1;
            }
            else {
                estimated_filter_ratio *= 2.2;
                grid_info_host[i].cnt = cnt;
                grid_info_host[i].len = ranges_len;
                max_grids_cnt /= cnt;
            }
        }
        
    }

    void index::build_first_level_index(
            build_input_t const& in
    )
    {
        uint64_t l = f_config.l;
        uint64_t n_data = in.dataset.extent(0);
        std::vector<grid_meta> grid_info_host(l);
        generate_grid_meta(in, grid_info_host);

        auto &clusters_list = secondary_index_data.first_level_index.clusters_list;

        std::vector<std::vector<float>> grid_start_arrays(l);
        std::vector<float> grid_len(l);
        uint64_t tot_grids = 1;

        for (uint64_t i = 0; i < l; i++) {
            float start = grid_info_host[i].start;
            float end = grid_info_host[i].end;
            float len = grid_info_host[i].len;
            uint64_t cnt = grid_info_host[i].cnt;

            grid_start_arrays[i].resize(cnt);
            
            uint64_t stride = len;
            if (len >= (end - start) * 0.5)
                stride = (end - start - len) / (cnt - 1);
            for (uint64_t j = 0; j < cnt; j++)
                grid_start_arrays[i][j] = start + j * stride;
            grid_len[i] = len;
            tot_grids *= cnt;
        }
        std::vector<float> ranges_host_arrays(tot_grids * 2 * l);

        clusters_list.resize(tot_grids);
        auto valid_indices_pool = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(1, n_data);
        auto select_indices_in_grid = [&](std::vector<float> const& grid_start, uint64_t counter) {
            std::vector<float> ranges_host(2 * l);
            for (int i = 0; i < l; i++) {
                ranges_host[2 * i] = grid_start[i];
                ranges_host[2 * i + 1] = grid_start[i] + grid_len[i];
                ranges_host_arrays[counter * 2 * l + 2 * i] = ranges_host[2 * i];
                ranges_host_arrays[counter * 2 * l + 2 * i + 1] = ranges_host[2 * i + 1];
            }
            auto ranges_device = parafilter_mmr::make_device_matrix_view<float, uint64_t>(1, 2 * l);
            cudaMemcpy(ranges_device.data_handle(), ranges_host.data(), 2 * l * sizeof(float), cudaMemcpyHostToDevice);
            uint64_t valid_cnt = filter_valid_data(
                in.dev_resources, 
                in.data_labels, 
                ranges_device,
                valid_indices_pool
            ); 
            
            clusters_list[counter] = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(valid_cnt);
            cudaMemcpy(clusters_list[counter].data_handle(), valid_indices_pool.data_handle(), sizeof(uint64_t) * valid_cnt, cudaMemcpyDeviceToDevice);
        };
        cartesian_product(grid_start_arrays, select_indices_in_grid);
        grid_params.ranges = parafilter_mmr::make_device_matrix_view<float, uint64_t>(tot_grids, 2 * l);
        cudaMemcpy(grid_params.ranges.data_handle(), ranges_host_arrays.data(), tot_grids * 2 * l * sizeof(float), cudaMemcpyHostToDevice);
    }
    void index::select_first_level_sets(
            query_input_t const& query_in, 
            raft::device_matrix_view<int, uint64_t> bitmap_matrix
    ) const
    {
        uint64_t n_queries = query_in.queries.extent(0);
        uint64_t l = f_config.l;
        uint64_t grids_cnt = secondary_index_data.first_level_index.clusters_list.size();

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((grids_cnt + 15) / 16, (n_queries + 15) / 16);

        auto filters = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, 2 * l);
        recovery_filters(query_in.query_labels, filters, f_config);

        intersect_kernel<<<numBlocks, threadsPerBlock>>>(
            filters.data_handle(), grid_params.ranges.data_handle(), bitmap_matrix.data_handle(),
            n_queries, grids_cnt, l
        );
    }
};