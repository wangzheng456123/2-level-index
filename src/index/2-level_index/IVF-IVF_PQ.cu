#include <index/2-level_index/IVF-IVF_PQ.cuh>

namespace IVF_IVF_PQ {
    static void __global__ count_valid_elements_atomic_kernel(const float* filter, const float* data_labels, 
                                        const uint64_t* indices, uint64_t l, uint64_t n_valid, 
                                        int* count, const uint64_t* sample_ids, int n_samples)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= n_samples) return;
        int iidx = sample_ids[x];
        uint64_t idx = indices[iidx];

        const float* cur_label = data_labels + idx * l;
        int valid = 1;
        for (int i = 0; i < l; i++) {
            float li = filter[i * 2];
            float ri = filter[i * 2 + 1];

            if (cur_label[i] < li || cur_label[i] > ri) {
                valid = 0;
                break;
            }
        }
        if (valid) {
            atomicAdd(count, uint64_t(1));
        }
    }

    __global__ static void check_valid_kernel(
                                            int* cnt_matrix,
                                            int n_samples,
                                            int* bitmap_matrix,
                                            int n_rows,
                                            int n_columns,
                                            float thresh_hold
                                        ) 
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < n_rows && col < n_columns) {
            int idx = row * n_columns + col;
            int count = cnt_matrix[idx];
            float ratio = static_cast<float>(count) / static_cast<float>(n_samples);
            bitmap_matrix[idx] = (ratio > thresh_hold) ? 1 : 0;
        }
    }

    static void select_valid_group(raft::device_matrix_view<uint64_t, uint64_t> const& selected_clusters, 
                        std::vector<raft::device_vector_view<uint64_t, uint64_t>> const& clusters_list, 
                        raft::device_matrix_view<float, uint64_t> const& ranges, 
                        raft::device_matrix_view<float, uint64_t> const& data_labels, 
                        int n_samples,
                        raft::device_matrix_view<int, uint64_t>& cnt_matrix)
    {
        uint64_t l = data_labels.extent(1);
        uint64_t n_data = data_labels.extent(0);
        uint64_t n_queries = selected_clusters.extent(0);
        uint64_t n_selected = selected_clusters.extent(1);
        uint64_t n_centers = cnt_matrix.extent(1);

        std::vector<uint64_t> selected_cluster_host(n_queries * n_selected);
        cudaMemcpy(selected_cluster_host.data(), selected_clusters.data_handle(), 
            n_queries * n_selected * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        uint64_t* sampled_ids = static_cast<uint64_t*>(parafilter_mmr::mem_allocator(sizeof(uint64_t) * n_samples));

        int n_streams = 256;
        cudaStream_t streams[256];
        for (int i = 0; i < n_streams; i++) {
            cudaStreamCreate(&streams[i]);
        }

        for (uint64_t i = 0; i < n_queries; i++) {
            uint64_t *cur_id = selected_cluster_host.data() + i * n_selected;
            for (uint64_t j = 0; j < n_selected; j++) {
                uint64_t cluster_id = cur_id[j];
                
                auto cur_cluster_ids = clusters_list[cluster_id];
                uint64_t cur_cluster_len = cur_cluster_ids.extent(0);
                uint64_t cur_samples = std::min(uint64_t(n_samples), cur_cluster_len);

                dim3 gridSize(256);
                dim3 blockSize((cur_samples + gridSize.x - 1) / gridSize.x);

                curandState* d_states = static_cast<curandState*>(parafilter_mmr::mem_allocator(cur_samples * sizeof(curandState)));
                init_rng_kernel<<<gridSize, blockSize>>>(d_states, time(NULL));
                fill_random_kernel<<<gridSize, blockSize>>>(sampled_ids, d_states, cur_cluster_len, cur_samples);
                
                count_valid_elements_atomic_kernel<<<gridSize, blockSize, 0, streams[j % n_streams]>>>(
                    ranges.data_handle() + 2 * l * i, 
                    data_labels.data_handle(), cur_cluster_ids.data_handle(), l, n_data, cnt_matrix.data_handle() + i * n_centers + cluster_id, 
                    sampled_ids, n_samples);
            }
        }

        dim3 block(16, 16);
        dim3 grid((n_queries + block.x - 1) / block.x,
                  (n_centers + block.y - 1) / block.y);

        check_valid_kernel<<<grid, block>>>(cnt_matrix.data_handle(),
                                            n_samples,
                                            cnt_matrix.data_handle(),
                                            n_queries,
                                            n_centers,
                                            0.5);
    }

    uint64_t index::estimate_filter_ratio(
        query_input_t const& query_in    
    ) const
    {
        // currently no good way to estimate filter ratio for cos clustering
        return 10;
    }

    void index::build_first_level_index(
        build_input_t const& in
    ) 
    {
        X_IVF_PQ::first_level_index_t &first_level_index = secondary_index_data.first_level_index;
        uint64_t n_row = in.dataset.extent(0);
        uint64_t l_dim = filters_and_labels.data_labels.extent(1);

        raft::cluster::kmeans::KMeansParams params;

        params.n_clusters = ivf_config.clusters0;
        params.tol = 1e-5;
        params.metric = raft::distance::DistanceType::CosineExpanded;

        auto inv_list = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(n_row);

        float interia;
        uint64_t niters;

        ivf_params.centers = parafilter_mmr::make_device_matrix_view<float, uint64_t>(params.n_clusters, l_dim);
        
        raft::cluster::kmeans::fit_predict<float, uint64_t>(
            in.dev_resources, 
            params, 
            filters_and_labels.data_labels, 
            std::nullopt, 
            ivf_params.centers, 
            inv_list,
            raft::make_host_scalar_view(&interia), 
            raft::make_host_scalar_view(&niters));

        
        first_level_index.clusters_list.resize(params.n_clusters);
        group_by_cluster_id(inv_list, first_level_index.clusters_list);
    }

    void index::select_first_level_sets(
        query_input_t const& query_in, 
        raft::device_matrix_view<int, uint64_t> bitmap_matrix
    ) const
    {
        uint64_t n_data = pq_index.codebook.extent(1);
        uint64_t n_queries = query_in.queries.extent(0);
        uint64_t n_dim = query_in.queries.extent(1);
        uint64_t l = f_config.l;

        auto& first_level_index = secondary_index_data.first_level_index;

        auto ranges = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, 2 * l);
        recovery_filters(query_in.query_labels, ranges, f_config);

        auto& data_labels = filters_and_labels.data_labels;

        int ivf_clusters[2];
        ivf_clusters[0] = ivf_config.clusters0;
        ivf_clusters[1] = secondary_config.clusters1;

        auto metric = raft::distance::DistanceType::CosineExpanded;

        auto normalized_ranges = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, l);
        float* global_min = static_cast<float*>(parafilter_mmr::mem_allocator(l));
        float* global_max = static_cast<float*>(parafilter_mmr::mem_allocator(l));
        normalize_ranges(ranges, normalized_ranges, global_min, global_max);

        auto first_centers_dis = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, ivf_clusters[0]);
        raft::distance::pairwise_distance(query_in.dev_resources, normalized_ranges, ivf_params.centers, first_centers_dis, metric);

        int recip_filter_ratio = 5;
        int n_centers_select = recip_filter_ratio;

        auto selected_centers_dis = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, n_centers_select);
        auto selected_centers_idx = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, n_centers_select);

        raft::matrix::select_k<float, uint64_t>(query_in.dev_resources, first_centers_dis, std::nullopt, selected_centers_dis, selected_centers_idx, false);

        auto cnt_matrix = parafilter_mmr::make_device_matrix_view<int, uint64_t>(n_queries, ivf_clusters[0]);
        int n_samples = 1000;
        select_valid_group(selected_centers_idx, first_level_index.clusters_list, ranges, data_labels, n_samples, cnt_matrix);
    }
};