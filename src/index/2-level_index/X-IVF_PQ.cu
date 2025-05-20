#include <index/2-level_index/X-IVF_PQ.cuh>
namespace X_IVF_PQ {
    __global__ void select_rows_by_flag_kernel(const int* flags, const uint64_t *prefix_sum, 
                                               uint64_t* selected_data, uint64_t n_data) 
    {
        uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= n_data) return;

        if (flags[x]) {
            uint64_t loc = prefix_sum[x];
            selected_data[loc - 1] = x;
        }
    }

    __global__ void merge_selected_kernel(
        const uint64_t* selected_indices,        
        uint64_t *const* secondary_clusters_ptr,  
        const uint64_t* secondary_clusters_len,   
        const uint64_t* row_offset,
        uint64_t* merged_output,                  
        uint64_t n_queries,
        uint64_t n_selected,
        uint64_t row_stride)
    {
        uint64_t qid = blockIdx.x * blockDim.x + threadIdx.x;
        uint64_t col = blockIdx.y * blockDim.y + threadIdx.y;
        if (qid >= n_queries || col >= row_stride) return;
    
        uint64_t* out_row = merged_output + qid * row_stride;
        uint64_t center_idx = 0;
    
        for (uint64_t i = n_selected - 1; i >= 0; --i) {
            if (col >= row_offset[qid * n_selected + i]) {
                center_idx = i;
                break;
            }
        }
        uint64_t cur_offset = row_offset[qid * n_selected + center_idx];

        uint64_t sel_idx = selected_indices[qid * n_selected + center_idx];

        const uint64_t* src = secondary_clusters_ptr[sel_idx];
        uint64_t len = secondary_clusters_len[sel_idx];

        if (col - cur_offset < len)
            out_row[col] = src[col - cur_offset];
        else out_row[col] = std::numeric_limits<uint64_t>::max();
    }

    __global__ void calc_indices_size_kernel(
        const uint64_t* selected_indices,        
        const uint64_t* secondary_clusters_len,   
        uint64_t* row_size,    
        uint64_t* row_offset,              
        uint64_t n_queries,
        uint64_t n_selected
    )
    {
        uint64_t qid = blockIdx.x * blockDim.x + threadIdx.x;
        if (qid >= n_queries) return;
    
        uint64_t tot_len = 0;
        for (uint64_t i = 0; i < n_selected; ++i) {
            uint64_t sel_idx = selected_indices[qid * n_selected + i];
            uint64_t len = secondary_clusters_len[sel_idx];
            row_offset[qid * n_selected + i] = tot_len;
            tot_len += len;
        }
        row_size[qid] = tot_len;
    }

    void merge_selected_indices(
        raft::device_matrix_view<uint64_t, uint64_t> const& selected_indices,
        raft::device_vector_view<uint64_t*, uint64_t> const& secondary_clusters_ptr,
        raft::device_vector_view<uint64_t, uint64_t> const& secondary_clusters_list_len,
        raft::device_matrix_view<uint64_t, uint64_t>& merged_indices)
    {
        uint64_t n_queries = selected_indices.extent(0);
        uint64_t n_selected = selected_indices.extent(1);
        
        int block = 256;
        int grid = (n_queries + block - 1) / block;

        auto row_len = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(n_queries);
        auto row_offset = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, n_selected);
        calc_indices_size_kernel<<<grid, block>>>(
            selected_indices.data_handle(), 
            secondary_clusters_list_len.data_handle(), 
            row_len.data_handle(), 
            row_offset.data_handle(), 
            n_queries,
            n_selected
        );

        uint64_t max_len = thrust::reduce(thrust::device, row_len.data_handle(), row_len.data_handle() + n_queries, 
                static_cast<uint64_t>(0), thrust::maximum<uint64_t>());
        merged_indices = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, max_len);
        cudaMemset(merged_indices.data_handle(), 0x3f, n_queries * max_len * sizeof(uint64_t));
        uint64_t row_stride = merged_indices.extent(1);  

        dim3 block_dim(8, 32);
        dim3 grid_dim((n_queries + block_dim.x - 1) / block_dim.x, 
            (max_len + block_dim.y - 1) / block_dim.y);
    
        merge_selected_kernel<<<grid_dim, block_dim>>>(
            selected_indices.data_handle(),
            secondary_clusters_ptr.data_handle(),
            secondary_clusters_list_len.data_handle(),
            row_offset.data_handle(), 
            merged_indices.data_handle(),
            n_queries,
            n_selected,
            row_stride);
        
        return ;
    }

    static void __global__ select_valid_secondary_centers_dis_kernel(const int* cnt_matrix, const uint64_t* inv_secondary_id_map,
                                                          float* dis, uint64_t n_queries,  
                                                          int n_secondary_centers, int n_centers)
    {
        uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= n_queries || y >= n_secondary_centers) return; 

        int cid = inv_secondary_id_map[y];
        // a magic to indicate build process may fail
        if (cid >= n_centers) { 
            dis[x * n_secondary_centers + y] = 123456;
        }
        
        int is_valid = cnt_matrix[x * n_centers + cid];

        if (!is_valid)
            dis[x * n_secondary_centers + y] = std::numeric_limits<float>::max();
    }

    static void __global__ pairwise_distance_with_map_kernel(
        const float* mat1, const float* mat2, const int* cnt_matrix, 
        const uint64_t* inv_secondary_id_map, float* dis, 
        uint64_t row1, uint64_t row2, uint64_t dim, uint64_t cnt_matrix_dim
    )
    {
        uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;

        int cid = inv_secondary_id_map[y];
        if (cid >= row2) { 
            dis[x * row2 + y] = 123456;
        }
        
        int is_valid = cnt_matrix[x * cnt_matrix_dim + cid];

        if (!is_valid)
            dis[x * row2 + y] = std::numeric_limits<float>::max();
        else {
            dis[x * row2 + y] = 0;
            for (uint64_t i = 0; i < dim; i++)
                dis[x * row2 + y] += pow(mat1[x * dim + i] - mat1[y * dim + i], 2);
        }
    }

    static void pairwise_distance_with_map(
        raft::device_matrix_view<float, uint64_t> const& mat1, 
        raft::device_matrix_view<float, uint64_t> const& mat2,
        raft::device_matrix_view<int, uint64_t> const& cnt_matrix, 
        raft::device_matrix_view<uint64_t, uint64_t> const& inv_secondary_id_map,
        raft::device_matrix_view<float, uint64_t> &out 
    )
    {
        assert(mat1.extent(1) == mat2.extent(1));
        uint64_t row1 = mat1.extent(0);
        uint64_t row2 = mat2.extent(0);
        uint64_t dim = mat1.extent(1); 
        uint64_t cnt_matrix_dim = cnt_matrix.extent(1);

        dim3 threads_per_block(16, 16);
        dim3 blocks_per_grid((row1 + threads_per_block.x - 1) / threads_per_block.x, 
                (row2 + threads_per_block.y - 1) / threads_per_block.y);

        
        pairwise_distance_with_map_kernel<<<threads_per_block, blocks_per_grid>>>(
            mat1.data_handle(), mat2.data_handle(), cnt_matrix.data_handle(), 
            inv_secondary_id_map.data_handle(), out.data_handle(), row1, row2, 
            dim, cnt_matrix_dim
        );
    }

    static void select_secondary_centers(
                             raft::device_resources const& dev_resources,
                             raft::device_matrix_view<int, uint64_t> const& cnt_matrix, 
                             raft::device_matrix_view<float, uint64_t> const& secondary_centers,
                             raft::device_vector_view<uint64_t, uint64_t> const& inv_secondary_id_map, 
                             raft::device_matrix_view<float, uint64_t> const& queries, 
                             raft::device_matrix_view<uint64_t, uint64_t> &selected_centers_id, 
                             raft::device_matrix_view<float, uint64_t> &selected_centers_dis)
    {
        uint64_t n_queries = queries.extent(0);
        uint64_t n_centers = cnt_matrix.extent(1);
        uint64_t n_secondary_centers = inv_secondary_id_map.extent(0);

        auto query_scondary_centers_dis = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, n_secondary_centers);
        auto metric = raft::distance::DistanceType::L2Expanded;
        raft::distance::pairwise_distance(dev_resources, queries, secondary_centers, query_scondary_centers_dis, metric);

        dim3 threads_per_block(16, 16);
        dim3 blocks_per_grid((n_queries + threads_per_block.x - 1) / threads_per_block.x, 
                (n_secondary_centers + threads_per_block.y - 1) / threads_per_block.y);

        select_valid_secondary_centers_dis_kernel<<<blocks_per_grid, threads_per_block>>>(cnt_matrix.data_handle(), 
            inv_secondary_id_map.data_handle(), query_scondary_centers_dis.data_handle(), n_queries, n_secondary_centers, n_centers);

        raft::matrix::select_k<float, uint64_t>(dev_resources, query_scondary_centers_dis, std::nullopt, selected_centers_dis, selected_centers_id, true);
    }

    static __global__ void label_data_with_value_kernel(int* flags, int val, const uint64_t* ids, uint64_t n_data)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= n_data) return;

        if (ids[x] == val) {
            flags[x] = 1;
        }
        else {
            flags[x] = 0;
        }
    }

    // A common helper for all derived class
    void index::group_by_cluster_id(
        raft::device_vector_view<uint64_t, uint64_t> const& cluster_ids, 
        std::vector<raft::device_vector_view<uint64_t, uint64_t>>& grouped_data
    ) 
    {
        uint64_t n_data = cluster_ids.extent(0);
        uint64_t n_clusters = grouped_data.size();

        int* tmp_flags = static_cast<int*>(parafilter_mmr::mem_allocator(n_data * sizeof(int)));
        uint64_t* prefix_sum = static_cast<uint64_t*>(parafilter_mmr::mem_allocator(n_data * sizeof(uint64_t)));

        dim3 threads_block_size(256);
        dim3 blocks_per_grid(n_data + threads_block_size.x - 1 / threads_block_size.x);

        uint64_t tot_valid = 0;
        for (int i = 0; i < n_clusters; i++) {
            cudaMemset(tmp_flags, 0, n_data * sizeof(int));

            label_data_with_value_kernel<<<blocks_per_grid, threads_block_size>>>(tmp_flags, i, cluster_ids.data_handle(), n_data);

            thrust::device_ptr<int> thrust_flags(tmp_flags);
            thrust::device_ptr<uint64_t> thrust_prefix_sum(prefix_sum);
            thrust::inclusive_scan(thrust::device, thrust_flags, thrust_flags + n_data, thrust_prefix_sum, thrust::plus<uint64_t>());
            uint64_t n_valid = thrust::reduce(thrust::device, thrust_prefix_sum, thrust_prefix_sum + n_data, 0, thrust::maximum<uint64_t>());
            if (n_valid == 0) continue;
            tot_valid += n_valid;

            // todo: allocate these mem from a local pool.
            auto cur_cluster_group = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(n_valid);
            select_rows_by_flag_kernel<<<blocks_per_grid, threads_block_size>>>(tmp_flags, prefix_sum, cur_cluster_group.data_handle(), n_data);
            grouped_data[i] = cur_cluster_group;
        }
        assert(tot_valid == n_data);
    }

    void index::select_indices(
            query_input_t const& query_in, 
            raft::device_matrix_view<int, uint64_t> const& bitmap_matrix,
            raft::device_matrix_view<uint64_t, uint64_t> &indices 
        ) const
    {
        auto &second_level_index = secondary_index_data.second_level_index;
        uint64_t n_data = filters_and_labels.data_labels.extent(0);
        uint64_t n_queries = query_in.queries.extent(0);
        uint64_t tot_clusters = second_level_index.inv_secondary_id_map.extent(0);
        
        uint64_t estimated_filter_ratio = estimate_filter_ratio(query_in);
        uint64_t n_candi = topk * estimated_filter_ratio;
        uint64_t n_list = secondary_config.n_list;
        uint64_t ivf_secondary_select = n_list * ((n_candi * tot_clusters + n_data - 1) / n_data);

        auto selected_secondary_centers_dis = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, ivf_secondary_select);
        auto selected_secondary_centers_idx = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, ivf_secondary_select);

        select_secondary_centers(query_in.dev_resources, bitmap_matrix, second_level_index.secondary_centers_list, 
            second_level_index.inv_secondary_id_map, 
            query_in.queries, selected_secondary_centers_idx, selected_secondary_centers_dis);
        
        merge_selected_indices(selected_secondary_centers_idx, second_level_index.secondary_clusters_ptr, 
                second_level_index.secondary_clusters_list_len, indices);
    }

    void index::build(build_input_t const& in) 
    {          
        build_pq(in);
        build_filters_and_labels(in);
        build_first_level_index(in);
        build_second_level_index(in);
    }

    void index::query(query_input_t const& in, query_output_t &out) const
    {
        uint64_t n_queries = in.queries.extent(0);
        uint64_t tot_sets = secondary_index_data.first_level_index.clusters_list.size();
        auto bitmap_matrix = parafilter_mmr::make_device_matrix_view<int, uint64_t>(n_queries, tot_sets);
        uint64_t n_data = filters_and_labels.data_labels.extent(0);

        select_first_level_sets(in, bitmap_matrix);
        
        raft::device_matrix_view<uint64_t, uint64_t> indices{};
        select_indices(in, bitmap_matrix, indices);
        
        valid_indices_t candidate_indices{}; 

        candidate_indices.valid_cnt = indices.extent(1);
        candidate_indices.indices = indices;

        uint64_t refine_ratio = pq_config.exps0;
        if (refine_ratio > 1) {
            query_output_t tmp_out;
            tmp_out.neighbors = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, refine_ratio * topk);
            tmp_out.distances = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, refine_ratio * topk);
            
            query_prefilter(in, candidate_indices, tmp_out);
            refine(in, tmp_out, out);
        }
        else query_prefilter(in, candidate_indices, out);
    }

    void index::build_second_level_index(
        build_input_t const& in
    ) 
    {   
        uint64_t& max_cluster_len = secondary_index_data.second_level_index.max_cluster_len;
        uint64_t& min_cluster_len = secondary_index_data.second_level_index.min_cluster_len;

        first_level_index_t &first_level_index = secondary_index_data.first_level_index;
        second_level_index_t &second_level_index = secondary_index_data.second_level_index;

        uint64_t n_dim = in.dataset.extent(1);
        uint64_t n_data = in.dataset.extent(0);
        min_cluster_len = n_data;
        max_cluster_len = 0;

        raft::cluster::kmeans::KMeansParams params;
        params.metric = raft::distance::DistanceType::L2Expanded;
        params.n_clusters = secondary_config.clusters1; 

        float interia;
        uint64_t niters;

        int n_clusters[2];

        n_clusters[0] = first_level_index.clusters_list.size();
        n_clusters[1] = params.n_clusters;

        int tot_centers = 0;
        uint64_t inv_offset = 0; 

        second_level_index.secondary_centers_list = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_clusters[0] * n_clusters[1], n_dim);
        second_level_index.inv_secondary_id_map = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(n_clusters[0] * n_clusters[1]);

        float* secondary_centers = static_cast<float*>(parafilter_mmr::mem_allocator(n_clusters[0] * n_clusters[1] * n_dim * sizeof(float)));
        uint64_t* inv_secondary_centers_ids = static_cast<uint64_t*>(parafilter_mmr::mem_allocator(n_clusters[0] * n_clusters[1] * sizeof(uint64_t)));

        uint64_t* inv_list = static_cast<uint64_t*>(parafilter_mmr::mem_allocator(n_data * sizeof(uint64_t)));

        uint64_t** secondary_clusters_ptr_host = new uint64_t*[n_clusters[0] * n_clusters[1]];
        uint64_t* secondary_clusters_list_len_host = new uint64_t[n_clusters[0] * n_clusters[1]];

        float* collected_vec_pool = static_cast<float*>(parafilter_mmr::mem_allocator(n_data * n_dim * sizeof(float)));
        uint64_t tot_ids = 0;
        for (int i = 0; i < n_clusters[0]; i++)
            tot_ids += first_level_index.clusters_list[i].extent(0);
        uint64_t* cluster_id_list_mem = static_cast<uint64_t*>(parafilter_mmr::mem_allocator(tot_ids * sizeof(uint64_t)));

        for (int i = 0; i < n_clusters[0]; i++) {
            uint64_t cur_cluster_len = first_level_index.clusters_list[i].extent(0);
            if (cur_cluster_len == 0) continue;
            params.n_clusters = std::min(uint64_t(n_clusters[1]), cur_cluster_len);
            LOG(INFO) << "build ivf pq index on girds: " << i << " with size: " << cur_cluster_len;

            auto cur_centers_view = raft::make_device_matrix_view<float, uint64_t>(secondary_centers + tot_centers * n_dim, 
                params.n_clusters, n_dim);
            auto cur_inv_idx_view = raft::make_device_vector_view<uint64_t, uint64_t>(inv_list + inv_offset, 
                cur_cluster_len);

            auto collected_vec_view = raft::make_device_matrix_view<float, uint64_t>(collected_vec_pool, cur_cluster_len, n_dim);
            auto indices_view = raft::make_device_matrix_view<uint64_t, uint64_t>(first_level_index.clusters_list[i].data_handle(), cur_cluster_len, 1);
            select_elements<float, uint64_t>(in.dev_resources, in.dataset, indices_view, collected_vec_view);
            cudaDeviceSynchronize();

            raft::cluster::kmeans::fit_predict<float, uint64_t>(
                in.dev_resources, 
                params, 
                collected_vec_view, 
                std::nullopt, 
                cur_centers_view, 
                cur_inv_idx_view,
                raft::make_host_scalar_view(&interia), 
                raft::make_host_scalar_view(&niters)
            );
            cudaDeviceSynchronize();

            std::vector<raft::device_vector_view<uint64_t, uint64_t>> cur_secondary_clusters_list;
            cur_secondary_clusters_list.resize(params.n_clusters);
            group_by_cluster_id(cur_inv_idx_view, cur_secondary_clusters_list);
            
            uint64_t cur_list_offset = 0;
            for (int j = 0; j < cur_secondary_clusters_list.size(); j++) {
                uint64_t secondary_center_len = cur_secondary_clusters_list[j].extent(0);
                secondary_clusters_ptr_host[tot_centers + j] = cluster_id_list_mem + inv_offset + cur_list_offset;
                auto cur_list_view = raft::make_device_vector_view<uint64_t, uint64_t>(secondary_clusters_ptr_host[tot_centers + j], 
                    secondary_center_len);
                if (secondary_center_len) 
                    select_elements(in.dev_resources, first_level_index.clusters_list[i], cur_secondary_clusters_list[j], 
                        cur_list_view);
                else fill(cur_centers_view.data_handle() + j * n_dim, sqrt(std::numeric_limits<float>::max() / n_dim), n_dim);
                secondary_clusters_list_len_host[tot_centers + j] = secondary_center_len;

                max_cluster_len = max(max_cluster_len, secondary_clusters_list_len_host[tot_centers + j]);
                min_cluster_len = min(min_cluster_len, secondary_clusters_list_len_host[tot_centers + j]);
                cur_list_offset += secondary_center_len;
            }
            fill(inv_secondary_centers_ids + tot_centers, uint64_t(i), params.n_clusters);
            inv_offset += cur_cluster_len;
            tot_centers += params.n_clusters;
        }

        second_level_index.secondary_centers_list = raft::make_device_matrix_view<float, uint64_t>(secondary_centers, tot_centers, n_dim);
        second_level_index.inv_secondary_id_map = raft::make_device_vector_view<uint64_t, uint64_t>(inv_secondary_centers_ids, tot_centers);

        second_level_index.secondary_clusters_ptr = parafilter_mmr::make_device_vector_view<uint64_t*, uint64_t>(tot_centers);
        second_level_index.secondary_clusters_list_len = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(tot_centers);

        cudaMemcpy(second_level_index.secondary_clusters_ptr.data_handle(), secondary_clusters_ptr_host, sizeof(uint64_t*) * tot_centers, cudaMemcpyHostToDevice);
        cudaMemcpy(second_level_index.secondary_clusters_list_len.data_handle(), secondary_clusters_list_len_host, 
                sizeof(uint64_t) * tot_centers, cudaMemcpyHostToDevice);
    }
}
