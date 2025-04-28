#include <index/parafiltering.cuh>

namespace parafiltering {
    void index::build(build_input_t const& in) 
    {
        normalized_labels.data_labels = in.data_labels;
        build_pq(in);
    }

    static __global__ void calculate_filter_dis_kernel(float* dis, const float* constrains, 
        const float* data_labels, uint64_t l, uint64_t n_queries, uint64_t n_data) 
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= n_queries || y >= n_data) return ;
        int ans = 1;

        for (int i = 0; i < l; i++) {
            float li = constrains[x * l * 2 + i * 2];
            float ri = constrains[x * l * 2 + i * 2 + 1];

            float label = data_labels[y * l + i];

            if (label < li || label > ri) {
                ans = 0;
            }
        }

        uint64_t idx = x * n_data + y;
        if (ans == 0) dis[idx] = 1e6;
        else dis[idx] = 0;  
    }

    static void calc_pairwise_filter_dis(raft::device_matrix_view<float, uint64_t> const &data_labels, 
        raft::device_matrix_view<float, uint64_t> const &constrains, 
        raft::device_matrix_view<float, uint64_t> filter_dis) 
    {
        uint64_t n_data = data_labels.extent(0);
        uint64_t l = data_labels.extent(1);

        uint64_t n_queries = constrains.extent(0);

        dim3 thread_block_size(16, 16);
        dim3 grid_block_size((n_queries + thread_block_size.x - 1) / thread_block_size.x, 
        (n_data + thread_block_size.y) / thread_block_size.y);

        calculate_filter_dis_kernel<<<grid_block_size, thread_block_size>>>(filter_dis.data_handle(), 
        constrains.data_handle(), data_labels.data_handle(), l, n_queries, n_data);
    }
    
    void index::query(query_input_t const& in, query_output_t &out) const
    {
        uint64_t n_data = pq_index.codebook.extent(1);
        uint64_t n_queries = in.queries.extent(1);

        uint64_t n_dim = in.dataset.extent(1);
        uint64_t l = f_config.l;

        uint64_t exps[2];
        exps[0] = pq_config.exps0;
        exps[1] = parafilter_config.exps1;

        float merge_rate = parafilter_config.merge_rate;
        auto& data_labels = normalized_labels.data_labels;

        auto ranges = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, 2 * l);
        recovery_filters(in.query_labels, ranges, f_config);

        auto vec_dis = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, n_data);

        valid_indices_t valid_indices;
        valid_indices.valid_cnt = -1;
        calc_pq_dis(in, vec_dis, valid_indices);

        auto label_dis = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, n_data); 
        calc_pairwise_filter_dis(data_labels, ranges, label_dis);
        auto dis = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, n_data);
        matrix_add_with_weights<float, uint64_t>(in.dev_resources, vec_dis, label_dis, dis, 1.f, merge_rate);

        auto first_candi_labels = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, topk * exps[0] * l);
        auto first_val = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, topk * exps[0]);
        auto first_idx = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, topk * exps[0]);

        raft::matrix::select_k<float, uint64_t>(in.dev_resources, dis, std::nullopt, first_val, first_idx, true, true);
        
        select_elements<float, uint64_t>(in.dev_resources, data_labels, first_idx, first_candi_labels);
        select_elements<float, uint64_t>(in.dev_resources, vec_dis, first_idx, first_val, false);

        auto second_indices = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, topk * exps[1]);
        filter_candi_by_labels(in.dev_resources, first_candi_labels, ranges, first_val, topk * exps[1], second_indices);

        auto second_indices_direct = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, topk * exps[1]);
        select_elements<uint64_t, uint64_t>(in.dev_resources, first_idx, second_indices, second_indices_direct, false);
        
        query_output_t tmp_out;
        tmp_out.neighbors = second_indices;
        
        refine(in, tmp_out, out);
    }
}