#include <index/sorting-index.cuh>
namespace sorting {
    static __global__ void copy_filtered_indices_kernel(const uint64_t* data_ranges, 
        const uint64_t* indices, 
        uint64_t* out_list, 
        uint64_t max_cnt, uint64_t n_queries, 
        int selected_dim, int l, uint64_t n_data)
    {
        uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= n_queries || y >= max_cnt) return;
        uint64_t left = data_ranges[selected_dim * n_queries + x];
        uint64_t right = data_ranges[(selected_dim + l) * n_queries + x];

        uint64_t cur = left + y;
        uint64_t out_idx = x * max_cnt + y;
        if (cur >= right) out_list[out_idx] = std::numeric_limits<uint64_t>::max();
        else out_list[out_idx] = indices[selected_dim * n_data + cur];
    }

    static void copy_filtered_indices(
        raft::device_matrix_view<uint64_t, uint64_t> const& data_ranges,
        raft::device_matrix_view<uint64_t, uint64_t> const& indices,
        raft::device_matrix_view<uint64_t, uint64_t> &out_list,
        int selected_dim) 
    {
        uint64_t n_queries = data_ranges.extent(1);
        uint64_t l = data_ranges.extent(0) / 2;
        uint64_t n_data = indices.extent(1);
        uint64_t max_cnt = out_list.extent(1);

        int full_block_per_grid_x = (n_queries + block_size_x - 1) / block_size_x;
        int full_block_per_grid_y = (max_cnt + block_size_y - 1) / block_size_y;

        dim3 full_blocks_per_grid(full_block_per_grid_x, full_block_per_grid_y);
        dim3 full_thread_per_grid(block_size_x, block_size_y);

        copy_filtered_indices_kernel<<<full_blocks_per_grid, full_thread_per_grid>>>(data_ranges.data_handle(), 
        indices.data_handle(), out_list.data_handle(), max_cnt, n_queries, selected_dim, l, n_data);
    }

    void index::sorting_data_labels(build_input_t const& in)
    {
        uint64_t l = in.data_labels.extent(1);
        uint64_t n_data = in.data_labels.extent(0);

        // Allocate memory for the transposed matrix (l x n_data)
        float* transposed_data = static_cast<float*>(parafilter_mmr::mem_allocator(n_data * l * sizeof(float)));
        transpose_matrix(in.data_labels.data_handle(), transposed_data, n_data, l);

        // Allocate memory for sorted values and indices
        sorting_index.values_view = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_data, l);
        sorting_index.indices_view = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_data, l);

        float* sorted_data_device = sorting_index.values_view.data_handle();
        uint64_t* sorted_indices_device = sorting_index.indices_view.data_handle();

        for (uint64_t i = 0; i < l; i++) {
            float* col_start = transposed_data + i * n_data;
            uint64_t* idx_start = sorted_indices_device + i * n_data;
            
            // Initialize indices
            thrust::device_ptr<uint64_t> idx_ptr(idx_start);
            thrust::sequence(idx_ptr, idx_ptr + n_data);

            // Sort values while keeping track of indices
            thrust::sort_by_key(thrust::device, col_start, col_start + n_data, idx_start);
            cudaMemcpy(sorted_data_device + i * n_data, col_start, sizeof(float) * n_data, cudaMemcpyDeviceToDevice);
        }
    }

    void index::build(build_input_t const&in) 
    {
        build_pq(in);
        build_filters_and_labels(in);
        sorting_data_labels(in);
    }

    void index::query_sorting_indices(query_input_t const &in, valid_indices_t &valid_indices) const
    {

        auto& sorted_data_labels_view = sorting_index.values_view;
        auto& sorted_data_labels_idx_view = sorting_index.indices_view;

        uint64_t n_queries = in.queries.extent(0);
        uint64_t n_data = sorted_data_labels_view.extent(0);
        uint64_t l = f_config.l;

        auto ranges = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, 2 * l);
        recovery_filters(in.query_labels, ranges, f_config);

        // Allocate memory for data_ranges on the device
        auto data_ranges_view = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(2 * l, n_queries);
        uint64_t* device_data_ranges = data_ranges_view.data_handle();
        float* ranges_host = new float[n_queries * 2 * l];
        uint64_t* data_ranges_host = new uint64_t[n_queries * 2 * l];

        auto shuffled_ranges = parafilter_mmr::make_device_matrix_view<float, uint64_t>(2 * l, n_queries);
        shuffle_data<float, uint64_t>(ranges, shuffled_ranges);
        auto shuffled_ranges_ptr = shuffled_ranges.data_handle();

        cudaMemcpy(ranges_host, ranges.data_handle(), n_queries * 2 * l * sizeof(float), cudaMemcpyDeviceToHost);
        uint64_t* max_len = new uint64_t[l];

        for (uint64_t i = 0; i < l; i++) {
            float* sorted_col = sorted_data_labels_view.data_handle() + i * n_data; // Sorted values for the i-th column
            uint64_t* sorted_idx_col = sorted_data_labels_idx_view.data_handle() + i * n_data; // Corresponding indices

            thrust::device_ptr<float> lower_queries_ptr(shuffled_ranges.data_handle() + i * n_queries);
            thrust::device_ptr<float> upper_queries_ptr(shuffled_ranges.data_handle() + (i + l) * n_queries);

            thrust::device_ptr<uint64_t> lower_indices_ptr(device_data_ranges + i * n_queries);
            thrust::device_ptr<uint64_t> upper_indices_ptr(device_data_ranges + (i + l) * n_queries);

            thrust::lower_bound(
            thrust::device,
                sorted_col, sorted_col + n_data,
                lower_queries_ptr, lower_queries_ptr + n_queries,
                lower_indices_ptr
            );

            thrust::upper_bound(
            thrust::device,
                sorted_col, sorted_col + n_data,
                upper_queries_ptr, upper_queries_ptr + n_queries,
                upper_indices_ptr
            );

            thrust::host_vector<uint64_t> h_lower(lower_indices_ptr, lower_indices_ptr + n_queries);
            thrust::host_vector<uint64_t> h_upper(upper_indices_ptr, upper_indices_ptr + n_queries);

            uint64_t max_range = 0;
            for (uint64_t j = 0; j < n_queries; j++) {
                uint64_t lower_idx = h_lower[j];
                uint64_t upper_idx = h_upper[j];
                uint64_t range_len = upper_idx - lower_idx;

                max_range = std::max(max_range, range_len);
            }

            max_len[i] = max_range;
        }

        uint64_t label_dim = 0;
        uint64_t min_cnt = 1e18;

        for (int i = 0; i < l; i++) {
            if (max_len[i] < min_cnt) {
                min_cnt = max_len[i];
                label_dim = i;
            }
        }

        uint64_t* coarsed_filterd_indices_pool = static_cast<uint64_t*>(parafilter_mmr::mem_allocator(n_queries * n_data * sizeof(uint64_t)));
        valid_indices.indices = raft::make_device_matrix_view<uint64_t, uint64_t>(coarsed_filterd_indices_pool, n_queries, min_cnt);
        valid_indices.valid_cnt = min_cnt;
        // Copy the data_ranges back to the device
        copy_filtered_indices(data_ranges_view, sorted_data_labels_idx_view, valid_indices.indices, label_dim);

        delete[] ranges_host;
        delete[] data_ranges_host;
    }

    void index::query(query_input_t const &in, query_output_t& out) const
    {
        uint64_t n_queries = in.queries.extent(0);

        valid_indices_t coarse_filterd_indices{};
        query_sorting_indices(in, coarse_filterd_indices);

        query_output_t tmp_out;
        tmp_out.neighbors = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, topk * pq_config.exps0);
        tmp_out.distances = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, topk * pq_config.exps0);
        query_prefilter(in, coarse_filterd_indices, tmp_out);

        refine(in, tmp_out, out);
    }
}