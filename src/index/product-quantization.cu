#include <index/product-quantization.cuh>

namespace pq {
    static __global__ void build_pq_lut_kernel(
        const float* centers, const float* queries,
        uint64_t query_batch_size, float* lut,
        uint64_t pq_len, uint64_t pq_dim, uint64_t n_dim,
        uint64_t n_queries, uint64_t n_clusters)
    {
        uint64_t query_batch_id = blockIdx.x * blockDim.x + threadIdx.x;
        uint64_t cluster_id = blockIdx.y * blockDim.y + threadIdx.y;
        uint64_t cur_dim = blockIdx.z * blockDim.z + threadIdx.z;

        if (cluster_id >= n_clusters || cur_dim >= pq_dim) return;

        for (uint64_t i = 0; i < query_batch_size; i++) {
            uint64_t qid = query_batch_id * query_batch_size + i;
            if (qid >= n_queries) return;

            uint64_t lut_index = qid * pq_dim * n_clusters + cur_dim * n_clusters + cluster_id;
            float ans = 0;

            for (uint64_t d = 0; d < pq_len; d++) {
                uint64_t query_index = qid * n_dim + cur_dim * pq_len + d;
                uint64_t center_index = cur_dim * pq_len * n_clusters + cluster_id * pq_len + d;

                if (cur_dim * pq_len + d >= n_dim) break;

                ans += (centers[center_index] - queries[query_index]) *
                    (centers[center_index] - queries[query_index]);
            }

            lut[lut_index] = ans;
        }
    }

    static __global__ void compute_batched_L2_distance_kernel(
        const uint8_t* codebook,   // Codebook: n_data * pq_dim, column-major
        const float* lut,        // LUT: n_queries * pq_dim * n_clusters
        const uint64_t* indices,      // Post filtered indices 
        float* result,           // Output: n_queries * n_data
        uint64_t n_data,              // Number of valid data after filter
        uint64_t tot_data,            // Total number of data
        uint64_t coarsed_data_cnt,    // Number of data after coarsed filter
        uint64_t pq_dim,              // Number of dimensions
        uint64_t n_clusters,          // Number of clusters
        uint64_t n_queries,           // Number of queries
        uint64_t data_batch_size,     // Batch size for data
        uint64_t query_batch_size   // Batch size for queries
        )    
    {
        // Thread's starting position
        uint64_t query_start = blockIdx.y * blockDim.y * query_batch_size + threadIdx.y * query_batch_size;
        uint64_t data_start = blockIdx.x * blockDim.x * data_batch_size + threadIdx.x * data_batch_size;

        // Temporary sum storage for batch
        for (uint64_t q = 0; q < query_batch_size && query_start + q < n_queries; q++) {
            for (uint64_t d = 0; d < data_batch_size && data_start + d < n_data; d++) {
                uint64_t data_index;
                if (indices != nullptr)
                    data_index = indices[(query_start + q) * coarsed_data_cnt + data_start + d];
                else data_index = data_start + d;
                if (data_index >= tot_data) {
                    result[(query_start + q) * n_data + data_start + d] = std::numeric_limits<float>::max();
                    continue;
                }
                float sum = static_cast<float>(0);

                for (uint64_t dim = 0; dim < pq_dim; dim++) {
                    uint8_t lut_idx = codebook[dim * tot_data + data_index];  // Column-major access
                    sum += lut[(query_start + q) * pq_dim * n_clusters + dim * n_clusters + lut_idx];
                }

                result[(query_start + q) * n_data + data_start + d] = sum;
            }
        }
    }

    // calculate similarity between batched datas and queries
    // here, codebook is row major with dim pq_dim * N, and centers is a matrix with (pq_len * N) * pq_dim matrix
    void index::calc_batched_L2_distance(
            raft::device_resources const& dev_resources,
            raft::device_matrix_view<float, uint64_t> const& queries,       
            raft::device_matrix_view<uint64_t, uint64_t> const& indices,
            // index_t pq_index,
            raft::device_matrix_view<float, uint64_t> dis,
            uint64_t query_batch_size, 
            uint64_t data_batch_size, 
            uint64_t n_clusters,
            int64_t n_indices
            ) const
    {
        uint64_t n_dim = queries.extent(1);
        uint64_t tot_data = pq_index.codebook.extent(1);
        uint64_t n_data = tot_data;
        uint64_t n_queries = queries.extent(0);
        uint64_t coarsed_data_cnt = indices.extent(1);

        uint64_t pq_dim = pq_config.pq_dim;  
        uint64_t pq_len = pq_config.pq_len;

        if (n_indices >= 0) {
            n_data = n_indices;
        }

        if (n_indices >= 0) {
            n_data = n_indices;
        }

        LOG(INFO) << "calc vector similarity: with data size: " << n_data 
                        << ", query cnt: " << n_queries << ", pq dimension: " << pq_dim << "\n";

        // int one cuda thread, process a batched of queries and data vectors
        uint64_t data_batch_cnt = (n_data + data_batch_size - 1) / data_batch_size;
        uint64_t query_batch_cnt = (n_queries + query_batch_size - 1)  / query_batch_size;

        float* lut;
        uint64_t lut_size = n_queries * pq_dim * n_clusters * sizeof(float);
        lut = (float *)parafilter_mmr::mem_allocator(lut_size);

        int lut_block_dim_z = pq_dim;
        int lut_block_dim_x = (n_queries + block_size_x - 1) / block_size_x;
        int lut_block_dim_y = (n_clusters + block_size_y - 1) / block_size_y;

        dim3 lut_full_blocks_per_grid(lut_block_dim_x, lut_block_dim_y, lut_block_dim_z);
        dim3 lut_full_threads_per_grid(block_size_x, block_size_y, 1);

        build_pq_lut_kernel<<<lut_full_blocks_per_grid, lut_full_threads_per_grid>>>(
            pq_index.centers.data_handle(), queries.data_handle(), query_batch_size, 
            lut, pq_len, pq_dim, n_dim, n_queries, n_clusters
        );
        checkCUDAErrorWithLine("launch lut build kernel failed!");

        uint64_t block_dim_x = (data_batch_cnt + block_size_x - 1) / block_size_x;
        uint64_t block_dim_y = (query_batch_cnt + block_size_y - 1) / block_size_y;

        dim3 full_blocks_per_grid(block_dim_x, block_dim_y);
        dim3 full_threads_per_block(block_size_x, block_size_y);

        compute_batched_L2_distance_kernel<<<full_blocks_per_grid, full_threads_per_block>>>(
            pq_index.codebook.data_handle(), lut, indices.data_handle(), dis.data_handle(), n_data, tot_data, 
            coarsed_data_cnt, pq_dim, n_clusters, n_queries, data_batch_size, query_batch_size
        );
        checkCUDAErrorWithLine("launch pq similarity calculation kernel failed!");
    }

    void index::build_pq(
        build_input_t const& in
    ) 
    {
        uint64_t pq_dim = pq_config.pq_dim;
        uint64_t n_clusters = pq_config.n_clusters;

        auto &dataset = in.dataset;

        auto& codebook = pq_index.codebook;
        auto& pq_centers = pq_index.centers;

        uint64_t n_row = dataset.extent(0);
        uint64_t n_dim = dataset.extent(1);
        uint64_t pq_len = (n_dim + pq_dim - 1) / pq_dim;
        pq_config.pq_len = pq_len;
        uint64_t centers_len = n_clusters * pq_len;

        codebook = parafilter_mmr::make_device_matrix_view<uint8_t, uint64_t>(pq_dim, n_row);
        pq_centers = parafilter_mmr::make_device_matrix_view<float, uint64_t>(pq_dim, centers_len);
        auto tmp_train = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_row, pq_len);
        auto tmp_labels = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(n_row);

        raft::cluster::kmeans::KMeansParams params;
        params.metric = raft::distance::DistanceType::L2Expanded;
        params.n_clusters = n_clusters;

        float interia;
        uint64_t niters;

        for (int i = 0; i < pq_dim; i++) {
            if (i == pq_dim - 1) {
                cudaMemset(tmp_train.data_handle(), 0, tmp_train.size());
            }
            slice_coordinates<uint64_t> coords{0, i * pq_len, n_row, std::min((i + 1) * pq_len, n_dim)};
            // todo: remove this copy since it is time consuming for large dataset.
            slice(in.dev_resources, dataset, tmp_train, coords);

            auto cur_centers_view = raft::make_device_matrix_view<float, uint64_t>(pq_centers.data_handle() + centers_len * i, n_clusters, pq_len);

            raft::cluster::kmeans::fit_predict<float, uint64_t>(
                                            in.dev_resources, 
                                            params, 
                                            tmp_train, 
                                            std::nullopt, 
                                            cur_centers_view, 
                                            tmp_labels,
                                            raft::make_host_scalar_view(&interia), 
                                            raft::make_host_scalar_view(&niters));

            auto tmp_quanted_vector_view = raft::make_device_vector_view<uint8_t, uint64_t>(codebook.data_handle() + i * n_row, n_row);

            raft::linalg::map_offset(in.dev_resources,  
                                    tmp_quanted_vector_view, 
                                    [] __device__ (const uint64_t idx, const uint64_t ele) {
                                        return static_cast<uint8_t>(ele);
                                    },
                                    raft::make_const_mdspan(tmp_labels));
        
        }
    }

    void index::build(
        build_input_t const& in
    ) 
    {
        build_pq(in);
    }

    void index::calc_pq_dis(const query_input_t& in, raft::device_matrix_view<float, uint64_t> pq_dis, valid_indices_t const& valid_indices) const
    {
        auto& queries = in.queries;
        auto& centers = pq_index.centers;
        auto& codebook = pq_index.codebook;
        auto pq_len = pq_config.pq_len;

        int n_data = codebook.extent(1);
        int n_queries = queries.extent(0);
        int n_dim = queries.extent(1);
        int n_clusters = centers.extent(1) / pq_len;

        if (valid_indices.valid_cnt >= 0)
            LOG(TRACE) << "query batch maximum valid indices cnt: " << valid_indices.valid_cnt;

        assert(!(valid_indices.valid_cnt > 0 && valid_indices.indices.data_handle() == nullptr));
        auto &indices = valid_indices.indices;
        auto valid_cnt = valid_indices.valid_cnt;

        calc_batched_L2_distance(in.dev_resources, queries, indices, pq_dis, 1, 1, n_clusters, valid_cnt);
    }

    void index::query_pq(const query_input_t &in,
                         const valid_indices_t& valid_indices,  
                         query_output_t &out) const
    { 
        uint64_t n_queries = in.queries.extent(0);
        uint64_t n_data = pq_index.codebook.extent(1);

        void* vec_dis_mem = parafilter_mmr::mem_allocator(n_queries * n_data * sizeof(float));
        auto vec_dis = raft::make_device_matrix_view<float, uint64_t>((float *)vec_dis_mem, n_queries, valid_indices.valid_cnt);
      
        calc_pq_dis(in, vec_dis, valid_indices);

        assert(out.neighbors.extent(1) == out.distances.extent(1));
        assert(out.neighbors.extent(0) == out.distances.extent(0));
        
        auto first_idx_indirect = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, out.neighbors.extent(1));
      
        raft::matrix::select_k<float, uint64_t>(in.dev_resources, vec_dis, std::nullopt, out.distances, first_idx_indirect, true, true);
        
        if (valid_indices.valid_cnt >= 0)
            select_elements<uint64_t, uint64_t>(in.dev_resources, valid_indices.indices, first_idx_indirect, out.neighbors, false);
        else out.neighbors = first_idx_indirect;
    }

    void index::query(
        const query_input_t &in, 
        query_output_t &out) const
    {   
        valid_indices_t valid_indices{};
        query_pq(in, valid_indices, out);
    }
}