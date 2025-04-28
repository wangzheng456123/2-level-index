#include <core/refine.cuh>

// Kernel to compute the L2 distance between queries and the selected dataset indices
__global__ void compute_l2_distances_kernel(
    const uint64_t* selected_indices_ptr, 
    const float* dataset_ptr, 
    const float* queries_ptr, 
    float* distances_ptr, 
    uint64_t n_data, 
    uint64_t n_queries, 
    uint64_t n_candi, 
    uint64_t n_dim)
{
    uint64_t query_idx = blockIdx.x * blockDim.x + threadIdx.x;  // Query index (rows of queries)
    uint64_t cand_idx = blockIdx.y * blockDim.y + threadIdx.y;   // Candidate index (columns of selected indices)

    if (query_idx < n_queries && cand_idx < n_candi) {
        uint64_t dataset_idx = selected_indices_ptr[query_idx * n_candi + cand_idx];  // Get the index from selected indices
        // Compute L2 distance
        float dist = 0.0f;
        float data_val = std::sqrt(std::numeric_limits<float>::max() / static_cast<uint64_t>(n_dim));
        for (uint64_t d = 0; d < n_dim; ++d) {
            if (dataset_idx < n_data) {
                data_val = dataset_ptr[dataset_idx * n_dim + d];
            } 
            float diff = data_val - queries_ptr[query_idx * n_dim + d];
            dist += pow(diff, 2);
        }
    
        distances_ptr[query_idx * n_candi + cand_idx] = dist;  // Store the computed distance
    }
}

void refine(raft::device_resources const& dev_resources,
            raft::device_matrix_view<float, uint64_t> const& dataset,
            raft::device_matrix_view<float, uint64_t> const& queries,
            raft::device_matrix_view<uint64_t, uint64_t> const& neighbor_candidates,
            raft::device_matrix_view<uint64_t, uint64_t> &indices, 
            raft::device_matrix_view<float, uint64_t> &distances)  
{
    uint64_t n_data = dataset.extent(0);
    uint64_t n_dim = dataset.extent(1);
    uint64_t n_queries = queries.extent(0);
    assert(indices.extent(0) == n_queries && distances.extent(0) == n_queries);
    assert(indices.extent(1) == distances.extent(1));

    uint64_t n_candi = neighbor_candidates.extent(1);
    uint64_t k = indices.extent(1);

    int full_blocks_per_grid_x = (n_queries + block_size_x - 1) / block_size_x;
    int full_blocks_per_grid_y = (n_candi + block_size_y - 1) / block_size_y;

    dim3 full_blocks_per_grid(full_blocks_per_grid_x, full_blocks_per_grid_y);
    dim3 full_threads_per_block(block_size_x, block_size_y);

    auto refine_dis = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, n_candi);
    compute_l2_distances_kernel<<<full_blocks_per_grid, full_threads_per_block>>>(neighbor_candidates.data_handle(), 
        dataset.data_handle(), queries.data_handle(), refine_dis.data_handle(), n_data, n_queries, n_candi, n_dim);
    
    auto refine_indices = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, k);
    raft::matrix::select_k<float, uint64_t>(dev_resources, refine_dis, std::nullopt, distances, refine_indices, true);

    select_elements<uint64_t, uint64_t>(dev_resources, neighbor_candidates, refine_indices, indices, false);
    LOG(INFO) << "cur query batch finished";
}