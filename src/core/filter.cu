#include <core/filter.cuh>

/*for the value in dis, if it pass the filter constrains, save its value, otherwise drop it*/
static __global__ void label_matrix_by_filter_kernel(float* mat, const float* ranges, 
                                              const float* labels, int l, int n_row, int n_dim) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= n_row || y >= n_dim) return ;
    int ans = 1;

    for (int i = 0; i < l; i++) {
        float li = ranges[x * l * 2 + i * 2];
        float ri = ranges[x * l * 2 + i * 2 + 1];

        float label = labels[x * l * n_dim + y * l + i];

        if (label < li || label > ri) {
            ans = 0;
            break;
        }
    }

    int idx = x * n_dim + y;
    if (ans == 0) mat[idx] = std::numeric_limits<float>::max(); 
}

/*for the value in dis, if it pass the filter constrains, save its value, otherwise drop it*/
static __global__ void label_matrix_by_filter_indices_kernel(
    const float* ranges, 
    const float* labels, int l, 
    int n_row, uint64_t n_data, int* res, 
    uint64_t* indices, 
    uint64_t n_dim) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= n_row || y >= n_dim) return ;
    int ans = 1;

    uint64_t idx = y;
    if (indices != nullptr)
        idx = indices[x * n_dim + y];
    if (idx >= n_data) return;

    for (int i = 0; i < l; i++) {
        float li = ranges[x * l * 2 + i * 2];
        float ri = ranges[x * l * 2 + i * 2 + 1];

        float label = labels[idx * l + i];

        if (label < li || label > ri) {
            ans = 0;
            break;
        }
    }

    res[x * n_dim + y] = ans; 
}

static __global__ void write_res_kernel(const int* valid_flags, const uint64_t* valid_flags_prefix_sum, 
                                 uint64_t* output, uint64_t* row_counts, uint64_t rows, uint64_t cols) 
{
    uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols) return;

    uint64_t idx = row * cols + col;

    if (valid_flags[idx]) {
        uint64_t pos = valid_flags_prefix_sum[idx]; // Compute output position
        output[row * cols + pos - 1] = col; // Store column index of valid elements
    }

    // The last prefix sum entry in each row gives the total count
    if (col == cols - 1) {
        row_counts[row] = valid_flags_prefix_sum[idx]; 
    }
}

static __global__ void process_selected_indices_kernel(const float* pq_dis, uint64_t* SelectedIndices, 
                                                       uint64_t n_queries, int n_candies) 
{
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; 
    uint64_t j = blockIdx.y * blockDim.y + threadIdx.y; 

    if (i >= n_queries || j >= n_candies) return;

    uint64_t idx = i * n_candies + j;
    float value = pq_dis[idx];

    if (value == std::numeric_limits<float>::max()) {
        SelectedIndices[idx] = std::numeric_limits<uint64_t>::max();
    }
}

void filter_candi_by_labels(
    raft::device_resources const& dev_resources,
    raft::device_matrix_view<float, uint64_t> const& candi_labels, 
    raft::device_matrix_view<float, uint64_t> const& constrains, 
    raft::device_matrix_view<float, uint64_t> const& pq_dis,
    int topk, 
    raft::device_matrix_view<uint64_t, uint64_t> fcandi) 
{
    uint64_t n_constrains = constrains.extent(0);
    uint64_t l = constrains.extent(1) / 2;
    uint64_t n_candi = candi_labels.extent(1) / l;
    
    int full_block_per_grid_x = (n_constrains + block_size_x - 1) / block_size_x;
    int full_block_per_grid_y = (n_candi + block_size_y - 1) / block_size_y;

    dim3 full_blocks_per_grid(full_block_per_grid_x, full_block_per_grid_y);
    dim3 full_thread_per_grid(block_size_x, block_size_y);

    label_matrix_by_filter_kernel<<<full_blocks_per_grid, full_thread_per_grid>>>(pq_dis.data_handle(), constrains.data_handle(), 
        candi_labels.data_handle(), l, n_constrains, n_candi
    );

    auto select_val = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_constrains, topk);

    raft::matrix::select_k<float, uint64_t>(dev_resources, pq_dis, std::nullopt, select_val, fcandi, true, true);

    full_blocks_per_grid.y = (topk + block_size_y - 1) / block_size_y;
    process_selected_indices_kernel<<<full_blocks_per_grid, full_thread_per_grid>>>(select_val.data_handle(), fcandi.data_handle(), 
        n_constrains, topk);
    
    return ;
}

uint64_t filter_valid_data(raft::device_resources const& dev_resources,
                           raft::device_matrix_view<float, uint64_t> const &data_labels, 
                           raft::device_matrix_view<float, uint64_t> const &constrains,
                           raft::device_matrix_view<uint64_t, uint64_t> &valid_indices, 
                           raft::device_matrix_view<uint64_t, uint64_t> const &coarse_filtered_indices, 
                           bool is_filtered
                           )
{
    uint64_t n_data = data_labels.extent(0);
    uint64_t n_coarse_filtered_indices = n_data;
    if (is_filtered) {
        n_coarse_filtered_indices = valid_indices.extent(1);
    }

    uint64_t* coarse_filtered_indices_data = nullptr;
    if (is_filtered) {
        coarse_filtered_indices_data = coarse_filtered_indices.data_handle();
    }

    uint64_t l = constrains.extent(1) / 2;
    uint64_t n_constrains = constrains.extent(0);

    dim3 thread_block_size(16, 16);
    dim3 grid_block_size((n_constrains + thread_block_size.x - 1) / thread_block_size.x, 
                         (n_coarse_filtered_indices + thread_block_size.y - 1) / thread_block_size.y);

    // Allocate memory for intermediate results
    // int* valid_flags_pool = static_cast<int*>(parafilter_mmr::mem_allocator(n_constrains * n_data * sizeof(int)));
    // uint64_t* valid_flags_prefix_sum_pool = static_cast<uint64_t*>(parafilter_mmr::mem_allocator(n_constrains * n_data * sizeof(uint64_t)));
    auto valid_flags_view = parafilter_mmr::make_device_matrix_view<int, uint64_t>(n_constrains, n_coarse_filtered_indices);
    auto valid_flags_prefix_sum_view = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_constrains, n_coarse_filtered_indices);

    int* valid_flags = valid_flags_view.data_handle();
    uint64_t* valid_flags_prefix_sum = valid_flags_prefix_sum_view.data_handle();

    uint64_t* row_counts = static_cast<uint64_t*>(parafilter_mmr::mem_allocator(n_constrains * sizeof(uint64_t)));

    // auto valid_indices_tmp_pool = 
    //    static_cast<uint64_t*>(parafilter_mmr::mem_allocator(n_constrains * n_data * sizeof(uint64_t)));
    auto valid_indices_tmp = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(valid_indices.extent(0), valid_indices.extent(1));

    // Initialize valid_indices to a very large value
    thrust::device_ptr<uint64_t> valid_indices_ptr(valid_indices_tmp.data_handle());
    thrust::fill(valid_indices_ptr, valid_indices_ptr + valid_indices.size(), std::numeric_limits<uint64_t>::max());

    // Call mark_valid_kernel
    label_matrix_by_filter_indices_kernel<<<grid_block_size, thread_block_size>>>(
                                                              constrains.data_handle(), 
                                                              data_labels.data_handle(), 
                                                              l, n_constrains, n_data, valid_flags, 
                                                              coarse_filtered_indices_data,
                                                              n_coarse_filtered_indices);

    matrix_scan<int, uint64_t, uint64_t>(valid_flags_view, valid_flags_prefix_sum_view);
    // Call write_res_kernel
    write_res_kernel<<<grid_block_size, thread_block_size>>>(valid_flags, 
                                                             valid_flags_prefix_sum, 
                                                             valid_indices_tmp.data_handle(), 
                                                             row_counts, n_constrains, n_coarse_filtered_indices);
    
    // Compute the maximum value in row_counts using Thrust
    thrust::device_ptr<uint64_t> row_counts_ptr(row_counts);
    uint64_t max_value = thrust::reduce(row_counts_ptr, row_counts_ptr + n_constrains, 0, thrust::maximum<uint64_t>());
    if (is_filtered)
        select_elements<uint64_t, uint64_t>(dev_resources, coarse_filtered_indices, valid_indices_tmp, valid_indices, false);
    else valid_indices = valid_indices_tmp;

    return max_value;
}

void label_matrix_by_filter(raft::device_matrix_view<float, uint64_t> const &labels, 
                            raft::device_matrix_view<float, uint64_t> const &ranges, 
                            raft::device_matrix_view<float, uint64_t> &matrix) 
{
    uint64_t n_dim = labels.extent(0);
    uint64_t l = ranges.extent(0) / 2;

    uint64_t n_row = ranges.extent(0);

    dim3 thread_block_size(16, 16);
    dim3 grid_block_size((n_row + thread_block_size.x - 1) / thread_block_size.x, 
                (n_dim + thread_block_size.y) / thread_block_size.y);

    label_matrix_by_filter_kernel<<<grid_block_size, thread_block_size>>>(matrix.data_handle(), 
            ranges.data_handle(), labels.data_handle(), l, n_row, n_dim);
}







