#include <core/merge_res.cuh>
__global__ void modify_blocks(uint64_t* d_matrix, uint64_t n_queries, int topk, uint64_t batch_size, 
                              float start, uint64_t data_batch_size) 
{
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y; 
    uint64_t block_idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < n_queries && block_idx < batch_size) {
        uint64_t block_start = row * (batch_size * topk) + block_idx * topk;
        float add_value = start + block_idx * data_batch_size;

        for (uint64_t i = 0; i < topk; ++i) {
            d_matrix[block_start + i] += add_value;
        }
    }
}

void merge_intermediate_result(raft::device_resources const& dev_resources,
                               const std::string &file_path, 
                               uint64_t batch_size, 
                               uint64_t data_batch_size, 
                               uint64_t n_queries, 
                               int topk, 
                               uint64_t start_offset, 
                               int device_id,
                               raft::device_matrix_view<float, uint64_t> merged_dis_view, 
                               raft::device_matrix_view<uint64_t, uint64_t> merged_idx_view) 
{
    std::vector<std::vector<float>> dis_matrices;
    std::vector<std::vector<uint64_t>> idx_matrices;

    std::string dis_file_path = file_path + "distances_" + std::to_string(device_id);
    std::string neigh_file_path = file_path + "neighbors_" + std::to_string(device_id);

    read_matrices_from_file(dis_file_path, n_queries, topk, batch_size, dis_matrices);
    read_matrices_from_file(neigh_file_path, n_queries, topk, batch_size, idx_matrices);

    auto dis_view = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, topk * batch_size);
    auto idx_view = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, topk * batch_size);
    merge_matrices_to_gpu(dis_matrices, dis_view.data_handle(), n_queries, topk, batch_size);
    merge_matrices_to_gpu(idx_matrices, idx_view.data_handle(), n_queries, topk, batch_size);

    dim3 block_dim(block_size_x, block_size_y); 
    dim3 grid_dim((batch_size + block_dim.x - 1) / block_dim.x, (n_queries + block_dim.y - 1) / block_dim.y);
    modify_blocks<<<grid_dim, block_dim>>>(idx_view.data_handle(), n_queries, topk, 
                    batch_size, start_offset, data_batch_size);

    auto merged_idx_indirect_view =  parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, topk);
    
    raft::matrix::select_k<float, uint64_t>(
        dev_resources, dis_view, std::nullopt, merged_dis_view, merged_idx_indirect_view, true
    );

    select_elements<uint64_t, uint64_t>(dev_resources, idx_view, 
                         merged_idx_indirect_view, merged_idx_view, false);
}