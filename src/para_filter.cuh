#pragma once

#include "parafilter_utils.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/matrix/select_k.cuh>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <assert.h>
#include <cmath>

const int block_size = 128;
const int block_size_x = 32;
const int block_size_y = 16;

INIT_PARAFILTER

template<typename ElmentType, typename IndexType>
__global__ void min_max_reduce_kernel(ElmentType *input, ElmentType* output, IndexType n, bool is_min = true)
{
    IndexType block_size = blockDim.x;
    IndexType thread_id = threadIdx.x;
    IndexType block_id = blockIdx.x;

    IndexType chunk_size = block_size * 2;
    IndexType block_start = block_id * chunk_size;
    IndexType left;  // holds index of left operand
    IndexType right; // holds index or right operand
    IndexType threads = block_size;
    for (IndexType stride = 1; stride < chunk_size; stride *= 2, threads /= 2)
    {
        left = block_start + thread_id * (stride * 2);
        right = left + stride;

        if (thread_id < threads 
            && right < n) 
        {
            if (is_min)
                input[left] = std::min(input[right], input[left]);
            else input[left] = std::max(input[right], input[left]);
        }
        __syncthreads();
    }

    if (!thread_id)
    {
        output[block_id] = input[block_start];
    }
}

template <typename m_t, typename idx_t = int>
RAFT_KERNEL slice(const m_t* src_d, idx_t lda, m_t* dst_d, idx_t x1, idx_t y1, idx_t x2, idx_t y2)
{
  idx_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx_t dm = x2 - x1, dn = y2 - y1;
  if (idx < dm * dn) {
    idx_t i = idx % dm, j = idx / dm;
    idx_t is = i + x1, js = j + y1;
    dst_d[idx] = src_d[is + js * lda];
  }
}

template <typename idx_t>
struct slice_coordinates {
  idx_t row1;  ///< row coordinate of the top-left point of the wanted area (0-based)
  idx_t col1;  ///< column coordinate of the top-left point of the wanted area (0-based)
  idx_t row2;  ///< row coordinate of the bottom-right point of the wanted area (1-based)
  idx_t col2;  ///< column coordinate of the bottom-right point of the wanted area (1-based)

  slice_coordinates(idx_t row1_, idx_t col1_, idx_t row2_, idx_t col2_)
    : row1(row1_), col1(col1_), row2(row2_), col2(col2_)
  {
  }
};

/*for the value in dis, if it pass the filter constrains, save its value, otherwise drop it*/
__global__ void filter_by_constrain_kernel(float* dis, const float* constrains, 
                                            const float* candi_label, int l, int n_queries, int n_candi) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= n_queries || y >= n_candi) return ;
    int ans = 1;

    for (int i = 0; i < l; i++) {
        float li = constrains[x * l * 2 + i * 2];
        float ri = constrains[x * l * 2 + i * 2 + 1];

        float label = candi_label[x * l * n_candi + y * l + i];

        if (label < li || label > ri) {
            ans = 0;
            break;
        }
    }

    int idx = x * n_candi + y;
    if (ans == 0) dis[idx] = std::numeric_limits<float>::max(); 
}

/**/
__global__ void calculate_filter_dis_kernel(float* dis, const float* constrains, 
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

// fixme: this kernel currently unused, but it may useful future because its satisify pseudocode in paper
__global__ void normalize_constrains_kernel(const float* constrains, int l, int n_cons, float* res) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n_cons) return ;

    for (int i = 0; i < l; i++) {
        float li = constrains[l * x * 2 + 2 * i + 0];
        float ri = constrains[l * x * 2 + 2 * i + 1];

        float len = std::max(static_cast<double>(ri - li), 1e-5);
        res[l * x + i] = (li + ri) / 
                            len;
    }
}

__global__ void denormalize_labels_kernel(const float* data, uint64_t n_data,
    const float* shift_val, const int* map_types, const float* interval_map, const int* div_values, 
    uint64_t l, float* out)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_data) return;

    for (uint64_t i = 0; i < l; ++i) {
        uint64_t idx = tid * l + i;

        if (map_types[i] == 0) {
            out[idx * 2] = data[idx] - max(shift_val[2 * i], 1e-7);
            out[idx * 2 + 1] = data[idx] + max(shift_val[2 * i + 1], 1e-7);
        }
        else if (map_types[i] == 1) {
            int val = data[idx] - 1;
            out[idx * 2] = interval_map[2 * val];
            out[idx * 2 + 1] = interval_map[2 * val + 1];
        }
        else if (map_types[i] == 2) {
            int val = data[idx];
            out[idx * 2] = val / div_values[i];
            out[idx * 2 + 1] = std::numeric_limits<float>::max();
        }
        else if (map_types[i] == 3) {
            out[2 * idx] = data[2 * idx];
            out[2 * idx + 1] = data[2 * idx + 1];
        }
    }
}

__global__ void normalize_ranges_labels_kernel(float* normalized_data_labels,
    uint64_t n_data, float* global_min, float* global_max, const float* ranges, uint64_t l)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_data) return;

    for (uint64_t i = 0; i < l; ++i) {
        uint64_t idx = tid * l + i;
        float left = ranges[idx * 2];
        float right = ranges[idx * 2 + 1];
        float midpoint = (left + right) / 2.0f;
        float coeff = 2.f / (right - left);
        coeff = 2.f / (global_max[i] - global_min[i]);
        normalized_data_labels[idx] = (midpoint - global_min[i]) * coeff;
    }
}

__global__ void normalize_data_labels_kernel(const float* data, uint64_t n_data,
    const float* maps_len, const float* shift_len, const int* map_types, const int* div_values, float* global_min,
    float* global_max, uint64_t l, float* out)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_data) return;

    for (uint64_t i = 0; i < l; ++i) {
        uint64_t idx = tid * l + i;
        float coeff = 0.f;

        if (map_types[i] == 0) {
            coeff = 2.f / shift_len[i];
        }
        else if (map_types[i] == 1) {
            int val = data[idx] - 1;
            coeff = 2.f / maps_len[val];
        }
        else if (map_types[i] == 1) {
            float val = data[idx] / div_values[i];
            coeff = 2.f / (global_max[i] - val);
        }

        coeff = 2.f / (global_max[i] - global_min[i]);
        out[idx] = (data[idx] - global_min[i]) * coeff;
    }
}

template <typename ElementType, typename IndexType>
__global__ void build_pq_lut_kernel(
    const ElementType* centers, const ElementType* queries,
    IndexType query_batch_size, ElementType* lut,
    IndexType pq_len, IndexType pq_dim, IndexType n_dim,
    IndexType n_queries, IndexType n_clusters)
{
    IndexType query_batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType cluster_id = blockIdx.y * blockDim.y + threadIdx.y;
    IndexType cur_dim = blockIdx.z * blockDim.z + threadIdx.z;

    if (cluster_id >= n_clusters || cur_dim >= pq_dim) return;

    for (IndexType i = 0; i < query_batch_size; i++) {
        IndexType qid = query_batch_id * query_batch_size + i;
        if (qid >= n_queries) return;

        IndexType lut_index = qid * pq_dim * n_clusters + cur_dim * n_clusters + cluster_id;
        ElementType ans = 0;

        for (IndexType d = 0; d < pq_len; d++) {
            IndexType query_index = qid * n_dim + cur_dim * pq_len + d;
            IndexType center_index = cur_dim * pq_len * n_clusters + cluster_id * pq_len + d;

            if (cur_dim * pq_len + d >= n_dim) break;

            ans += (centers[center_index] - queries[query_index]) *
                 (centers[center_index] - queries[query_index]);
        }

        lut[lut_index] = ans;
    }
}

template <typename CodebookType, typename ElementType, typename IndexType>
__global__ void compute_batched_L2_distance_kernel(
    const CodebookType* codebook,   // Codebook: n_data * pq_dim, column-major
    const ElementType* lut,        // LUT: n_queries * pq_dim * n_clusters
    ElementType* result,           // Output: n_queries * n_data
    IndexType n_data,              // Number of data points
    IndexType pq_dim,              // Number of dimensions
    IndexType n_clusters,          // Number of clusters
    IndexType n_queries,           // Number of queries
    IndexType data_batch_size,     // Batch size for data
    IndexType query_batch_size)    // Batch size for queries
{
    // Thread's starting position
    IndexType query_start = blockIdx.y * blockDim.y * query_batch_size + threadIdx.y * query_batch_size;
    IndexType data_start = blockIdx.x * blockDim.x * data_batch_size + threadIdx.x * data_batch_size;

    // Temporary sum storage for batch
    for (IndexType q = 0; q < query_batch_size && query_start + q < n_queries; q++) {
        for (IndexType d = 0; d < data_batch_size && data_start + d < n_data; d++) {
            ElementType sum = static_cast<ElementType>(0);

            for (IndexType dim = 0; dim < pq_dim; dim++) {
                CodebookType lut_idx = codebook[dim * n_data + data_start + d];  // Column-major access
                sum += lut[(query_start + q) * pq_dim * n_clusters + dim * n_clusters + lut_idx];
            }

            result[(query_start + q) * n_data + data_start + d] = sum;
        }
    }
}

template <typename T, typename IndexType>
__global__ void transpose_kernel(const T* __restrict__ input, T* __restrict__ output,
                                  IndexType rows, IndexType cols) {
    // Thread and block indices
    IndexType x = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType y = blockIdx.y * blockDim.y + threadIdx.y;

    // Transpose the matrix directly using global memory
    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

// select a batch of idices of row from the input matrix
template<typename ElementType, typename IndexType>
__global__ void select_elements_kernel(const ElementType* input, const IndexType* indices, 
                                        ElementType *output, IndexType n_row_o, IndexType n_row_i, 
                                        IndexType n_dim_o, IndexType n_dim_i, bool is_select_row) 
{
    IndexType x = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= n_row_o || y >= n_dim_o) return ;
    IndexType idx = x;
    IndexType idy = indices[x * n_dim_o + y]; 
    ElementType max_value = std::numeric_limits<ElementType>::max();

    if (is_select_row) {
        for (IndexType i = 0; i < n_dim_i; i++) {
            IndexType o_idx = x * n_dim_i * n_dim_o + y * n_dim_i + i;
            if (idy < n_row_i) 
                output[o_idx] = input[n_dim_i * idy + i];
            else output[o_idx] = std::sqrt(max_value / static_cast<ElementType>(n_dim_i));
        }
    }
    else {
        IndexType o_idx = x * n_dim_o + y;
        if (idy < n_dim_i)
            output[o_idx] = input[n_dim_i * idx + idy];
        else output[o_idx] = max_value;
    }
}

// add C = w1 * A + w2 * B with matrix A, B, C
template<typename ElementType, typename IndexType>
__global__ void matrix_weight_add_kernel(const ElementType *A, const ElementType *B, ElementType *C,
                                       IndexType n_row, IndexType n_dim, ElementType w1, ElementType w2) 
{
    IndexType x = blockDim.x * blockIdx.x + threadIdx.x;
    IndexType y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= n_row || y >= n_dim) return ;

    IndexType idx = x * n_dim + y;
    C[idx] = w1 * A[idx] + w2 * B[idx]; 
} 

template<typename ElementType, typename IndexType>
__device__ ElementType calc_vector_L2_dis_device(const ElementType *v1, const ElementType *v2, IndexType n_dim)
{
    ElementType ans = 0;
    for (IndexType i = 0; i < n_dim; i++) {
        ans += pow(v1[i] - v2[i], 2);
    }

    return ans;
}

template<typename ElementType, typename IndexType>
__global__ void calc_refine_distance_kernel(const ElementType *candi_vec, const ElementType *queries, IndexType n_dim, 
                                            IndexType n_queries, IndexType n_candi, ElementType* dis) 
{
    IndexType x = blockDim.x * blockIdx.x + threadIdx.x;
    IndexType y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= n_queries || y >= n_candi) return;

    IndexType candi_idx = x * n_candi * n_dim + y * n_dim;
    IndexType query_idx = x * n_dim;   
    IndexType o_idx = x * n_candi + y;

    dis[o_idx] = calc_vector_L2_dis_device(candi_vec + candi_idx, queries + query_idx, n_dim);
}

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
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;  // Query index (rows of queries)
    int cand_idx = blockIdx.y * blockDim.y + threadIdx.y;   // Candidate index (columns of selected indices)

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

template <typename ElementType, typename IndexType>
__global__ void process_selected_indices_kernel(const ElementType* pq_dis, IndexType* SelectedIndices, 
                                                IndexType n_queries, int n_candies) 
{
    IndexType i = blockIdx.x * blockDim.x + threadIdx.x; 
    IndexType j = blockIdx.y * blockDim.y + threadIdx.y; 

    if (i >= n_queries || j >= n_candies) return;

    IndexType idx = i * n_candies + j;
    ElementType value = pq_dis[idx];

    if (value == std::numeric_limits<ElementType>::max()) {
        SelectedIndices[idx] = std::numeric_limits<IndexType>::max();
    }
}

__global__ void label_data_with_value_kernel(int* flags, int val, const uint64_t* ids, uint64_t n_data)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n_data) return;

    if (lds[x] == val) {
        flags[x] = 1;
    }
    else {
        flags[x] = 0;
    }
}

inline void filter_candi_by_labels(raft::device_resources const& dev_resources,
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

    filter_by_constrain_kernel<<<full_blocks_per_grid, full_thread_per_grid>>>(pq_dis.data_handle(), constrains.data_handle(), 
        candi_labels.data_handle(), l, n_constrains, n_candi
    );

    auto select_val = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_constrains, topk);

    raft::matrix::select_k<float, uint64_t>(dev_resources, pq_dis, std::nullopt, select_val, fcandi, true, true);

    full_blocks_per_grid.y = (topk + block_size_y - 1) / block_size_y;
    process_selected_indices_kernel<<<full_blocks_per_grid, full_thread_per_grid>>>(select_val.data_handle(), fcandi.data_handle(), 
        n_constrains, topk);
    
    return ;
}

template<typename IndexType> 
__global__ void modify_data_patch_offset_kernel(IndexType* indices, 
                                     IndexType last_offset, 
                                     IndexType stride, 
                                     IndexType n_row, 
                                     IndexType n_dim, 
                                     IndexType topk) 
{
    IndexType x = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n_row || y >= n_dim) return;

    IndexType batch_cnt = n_dim / topk;
    IndexType batch_id = y / topk;

    IndexType id = x * n_dim + y;
    
    indices[id] += last_offset - (batch_cnt - batch_id - 1) * stride;
} 

template <typename ElementType, typename IndexType>
__global__ void modify_blocks(ElementType* d_matrix, IndexType n_queries, int topk, IndexType batch_size, 
                              ElementType start, IndexType data_batch_size) 
{
    IndexType row = blockIdx.y * blockDim.y + threadIdx.y; 
    IndexType block_idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < n_queries && block_idx < batch_size) {
        IndexType block_start = row * (batch_size * topk) + block_idx * topk;
        ElementType add_value = start + block_idx * data_batch_size;

        for (IndexType i = 0; i < topk; ++i) {
            d_matrix[block_start + i] += add_value;
        }
    }
}

template <typename ElementType, typename IndexType>
__global__ void select_rows_by_flag_kernel(const int* flags, const IndexType *prefix_sum, 
                                           IndexType *selected_data, IndexType n_data) 
{
    IndexType x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n_data) return;

    if (flags[x]) {
        IndexType loc = prefix_sum[x];
        selected_data[loc * n_dim + i] = x;
    }
}

void count_valid_elements_atomic_kernel(const float* filter, const float* data_labels, 
                                        const uint64_t* indices, uint64_t l, uint64_t n_valid, 
                                        int* count, const int* sample_ids, int n_samples)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n_samples) return;
    int iidx = sample_ids[x];
    uint64_t idx = indices[iidx];

    const float* cur_label = data_labels + idx * l;
    int valid = 1;
    for (i = 0; i < l; i++) {
        float li = filter[i * 2];
        float ri = filter[i * 2 + 1];

        if (cur_label[i] < li || cur_label[i] > ri) {
            valid = 0;
            break;
        }
    }
    if (valid) {
        atomicAdd(count, 1);
    }
}

void __global__ select_valid_secondary_centers_dis_kernel(const int* cnt_matrix, const int* inv_secondary_id_map,
                                                          float* dis, int n_samples, float thresh_hold, uint64_t n_queries,  
                                                          int n_secondary_centers, int n_centers)
{
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= n_queries || y >= n_secondary_centers) return; 

    int cid = inv_secondary_id_map[i];
    // a magic to indicate build process may fail
    if (cid >= n_centers) { 
        dis[x * n_secondary_centers + y] = 123456;
    }
    
    int valid_cnt = cnt_matrix[cid];

    if (static_cast<float>(valid_cnt) / n_samples >= thresh_hold)
        dis[x * n_secondary_centers + y] = std::numeric_limits<float>::max();
}

template <typename T>
__global__ void fill_kernel(T* data, T value, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = value;
}

__global__ void init_rng_kernel(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void fill_random_kernel(int* out, curandState* states, int range, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int r = curand(&states[idx]) % range;
        out[idx] = r;
    }
}

__global__ void merge_selected_kernel(
    const uint64_t* selected_indices,        
    const uint64_t** secondary_clusters_ptr,  
    const uint64_t* secondary_clusters_len,   
    uint64_t* merged_output,                  
    uint64_t n_queries,
    uint64_t n_selected,
    uint64_t row_stride)
{
    uint64_t qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= n_queries) return;

    uint64_t write_offset = 0;
    uint64_t* out_row = merged_output + qid * row_stride;

    for (uint64_t i = 0; i < n_selected; ++i) {
        uint64_t sel_idx = selected_indices[qid * n_selected + i];
        const uint64_t* src = secondary_clusters_ptr[sel_idx];
        uint64_t len = secondary_clusters_len[sel_idx];

        for (uint64_t j = 0; j < len; ++j)
            out_row[write_offset + j] = src[j];

        write_offset += len;
    }
}

template <typename T, typename IndexType>
void transpose_matrix(const T* input, T* output, IndexType rows, IndexType cols) {
    // Define block and grid sizes
    if (cols == 1)  {
        cudaMemcpy(output, input, rows * cols * sizeof(T), cudaMemcpyDeviceToDevice);
        return ;
    }
    const int TILE_DIM = 32;
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);

    // Launch the kernel
    transpose_kernel<T, IndexType><<<gridDim, blockDim>>>(input, output, rows, cols);

    // Synchronize the stream (optional, for error checking)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
}

template <typename ElementType, typename IndexType>
inline ElementType array_min_max_reduce(ElementType* in_array_device,
    IndexType n_elements,
    bool is_min = true) {
    // Constants
    constexpr IndexType block_size = 256; // Adjust based on GPU architecture
    IndexType threads_cnt = n_elements;

    // Compute initial number of blocks
    IndexType block_cnt = (threads_cnt + 2 * block_size - 1) / (2 * block_size);
    IndexType remaining = n_elements;

    // Allocate temporary memory on the device
    ElementType* sums_device = nullptr;
    cudaMalloc(&sums_device, block_cnt * sizeof(ElementType));
    ElementType* in_array_temp = nullptr;
    cudaMalloc(&in_array_temp, n_elements * sizeof(ElementType));

    // Copy the input array to temporary memory
    cudaMemcpy(in_array_temp, in_array_device, n_elements * sizeof(ElementType), cudaMemcpyDeviceToDevice);

    ElementType final_result;

    // Iterative reduction
    while (remaining > 1) {
        // Launch reduction kernel
        min_max_reduce_kernel << <block_cnt, block_size >> > (in_array_temp, sums_device, remaining, is_min);
        cudaDeviceSynchronize();

        // Update remaining elements and block count
        remaining = block_cnt;
        block_cnt = (remaining + 2 * block_size - 1) / (2 * block_size);

        // Swap input and output arrays for the next iteration if needed
        if (remaining > 1) {
            std::swap(in_array_temp, sums_device);
        }
    }

    // Copy the final result back to the host
    cudaMemcpy(&final_result, sums_device, sizeof(ElementType), cudaMemcpyDeviceToHost);

    // Free allocated device memory
    cudaFree(sums_device);
    cudaFree(in_array_temp);

    return final_result;
}

template<typename ElementType, typename IndexType>
inline void preprocessing_labels(raft::device_resources const& dev_resources,
                                 raft::device_matrix_view<ElementType, IndexType> data_labels, 
                                 raft::device_matrix_view<ElementType, IndexType> normalized_data_labels,
                                 raft::device_matrix_view<ElementType, IndexType> query_labels, 
                                 raft::device_matrix_view<ElementType, IndexType> normalized_query_labels,
                                 raft::device_matrix_view<ElementType, IndexType> denormalized_query_labels,
                                 ElementType* global_min, 
                                 ElementType* global_max,
                                 const filter_config& f_config, 
                                 bool is_query_changed = true,
                                 bool is_data_changed = true, 
                                 bool reconfig = false
                                 ) 
{
    IndexType n_data = normalized_data_labels.extent(0);
    IndexType n_queries = normalized_query_labels.extent(0);

    IndexType l_dim = normalized_data_labels.extent(1);

    thread_local float* shift_val_dev = nullptr; 
    thread_local float* ranges_map_dev = nullptr; 
    thread_local float* shift_len_dev = nullptr;
    thread_local float* maps_len_dev = nullptr;
    thread_local int* div_value_dev = nullptr;

    thread_local int* map_types_dev = nullptr;
    thread_local bool is_configed_dev = false;

    thread_local ElementType* global_min_dev = nullptr;
    thread_local ElementType* global_max_dev = nullptr;

    if (!is_configed_dev || reconfig) {
        auto trans_vec_to_device = [](void*& dev_ptr, const void* src, size_t size) {
            dev_ptr = parafilter_mmr::mem_allocator(size);
            cudaMemcpy(dev_ptr, src, size, cudaMemcpyHostToDevice);
        };
        trans_vec_to_device(reinterpret_cast<void*&>(shift_val_dev), f_config.shift_val.data(), f_config.shift_val.size() * sizeof(float));

        std::vector<float> ranges_map;
        for (const auto& maps : f_config.interval_map) {
            for (int value : maps) {
                ranges_map.push_back(static_cast<float>(value));
            }
        }
        trans_vec_to_device(reinterpret_cast<void*&>(ranges_map_dev), ranges_map.data(), ranges_map.size() * sizeof(float));

        std::vector<float> shift_len;
        std::vector<std::vector<float>> maps_len;
        process_filter_config(f_config, shift_len, maps_len);

        trans_vec_to_device(reinterpret_cast<void*&>(shift_len_dev), shift_len.data(), shift_len.size() * sizeof(float));

        std::vector<float> maps_len_flat;
        for (const auto& map : maps_len) {
            maps_len_flat.insert(maps_len_flat.end(), map.begin(), map.end());
        }
        trans_vec_to_device(reinterpret_cast<void*&>(maps_len_dev), maps_len_flat.data(), maps_len_flat.size() * sizeof(float));
        trans_vec_to_device(reinterpret_cast<void*&>(map_types_dev), f_config.filter_type.data(), f_config.filter_type.size() * sizeof(int));
        trans_vec_to_device(reinterpret_cast<void*&>(div_value_dev), f_config.div_value.data(), f_config.div_value.size() * sizeof(int));

        trans_vec_to_device(reinterpret_cast<void*&>(global_min_dev), global_min, f_config.l * sizeof(float));
        trans_vec_to_device(reinterpret_cast<void*&>(global_max_dev), global_max, f_config.l * sizeof(float));

        is_configed_dev = true;
    }

    int full_block_per_grid_x = (n_data + block_size - 1) / block_size;
    dim3 full_block_per_grid(full_block_per_grid_x);

    // todo fuse the 3 kernel calls to 1
    if (is_data_changed) {
        
    }

    if (is_query_changed) {
        // Set grid configuration
        full_block_per_grid.x = (n_queries + block_size - 1) / block_size;

        // Call denormalize_labels_kernel
        denormalize_labels_kernel<<<full_block_per_grid, block_size>>>(
            query_labels.data_handle(),           // Input raw query labels
            n_queries,                            // Number of queries
            shift_val_dev,                        // Device pointer for shift values
            map_types_dev,                        // Device pointer for map types
            ranges_map_dev,                       // Device pointer for interval map
            div_value_dev,                        // Device pointer for div value
            f_config.l,                           // Length of intervals
            denormalized_query_labels.data_handle() // Output to denormalized_query_labels
        );

        // Call normalize_ranges_labels_kernel to replace normalize_labels_kernel
        normalize_ranges_labels_kernel<<<full_block_per_grid, block_size>>>(
            normalized_query_labels.data_handle(),           // Output to normalized_data_labels
            n_queries,                                       // Number of data points
            global_min_dev,                                  // Global minimum value
            global_max_dev,                                  // Global maximum value
            denormalized_query_labels.data_handle(),         // Device pointer for ranges
            f_config.l                                       // Length of intervals
        );
    }
}

// calculate similarity between batched datas and queries
// here, codebook is row major with dim pq_dim * N, and centers is a matrix with (pq_len * N) * pq_dim matrix
// todo: modify it to template
inline void calc_batched_L2_distance(raft::device_resources const& dev_resources,
                                     raft::device_matrix_view<float, uint64_t> const& queries, 
                                     raft::device_matrix_view<uint8_t, uint64_t> const& codebook,
                                     raft::device_matrix_view<float, uint64_t> const& centers, 
                                     raft::device_matrix_view<float, uint64_t> dis,
                                     uint64_t pq_dim,  
                                     uint64_t pq_len,
                                     uint64_t query_batch_size = 1, 
                                     uint64_t data_batch_size = 1, 
                                     uint64_t n_clusters = 256) 
{
    uint64_t n_dim = queries.extent(1);
    uint64_t n_data = codebook.extent(1);
    uint64_t n_queries = queries.extent(0);

    std::cout << "calc vector similarity: with data size: " << n_data 
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
        centers.data_handle(), queries.data_handle(), query_batch_size, lut, pq_len, pq_dim, n_dim, n_queries, n_clusters
    );
    checkCUDAErrorWithLine("launch lut build kernel failed!");

    uint64_t block_dim_x = (data_batch_cnt + block_size_x - 1) / block_size_x;
    uint64_t block_dim_y = (query_batch_cnt + block_size_y - 1) / block_size_y;

    dim3 full_blocks_per_grid(block_dim_x, block_dim_y);
    dim3 full_threads_per_block(block_size_x, block_size_y);

    compute_batched_L2_distance_kernel<<<full_blocks_per_grid, full_threads_per_block>>>(
        codebook.data_handle(), lut, dis.data_handle(), n_data, pq_dim, n_clusters, n_queries, data_batch_size, query_batch_size
    );
    checkCUDAErrorWithLine("launch pq similarity calculation kernel failed!");
} 

// select elements of in an indices set from the input matrix and put it into the output matrix
template<typename ElementType, typename IndexType>
inline void select_elements(raft::device_resources const& dev_resources,
                            raft::device_matrix_view<ElementType, IndexType> const& input_matrix,
                            raft::device_matrix_view<IndexType, IndexType> const& indices, 
                            raft::device_matrix_view<ElementType, IndexType> output,
                            bool is_select_row = true) 
{
    IndexType n_row_o = indices.extent(0);
    IndexType n_row_i = input_matrix.extent(0);
    
    IndexType n_dim_i = input_matrix.extent(1);
    IndexType n_dim_o = indices.extent(1); 

    int full_block_per_grid_x = (n_row_o + block_size_x - 1) / block_size_x;
    int full_block_per_grid_y = (n_dim_o + block_size_y - 1) / block_size_y;

    dim3 full_blocks_per_grid(full_block_per_grid_x, full_block_per_grid_y);
    dim3 full_threads_per_block(block_size_x, block_size_y);

    select_elements_kernel<<<full_blocks_per_grid, full_threads_per_block>>>(input_matrix.data_handle(), 
            indices.data_handle(), output.data_handle(), n_row_o, n_row_i, n_dim_o, n_dim_i, is_select_row);

}

template<typename ElementType, typename IndexType>
inline void matrix_add_with_weights(raft::device_resources const& dev_resources,
                                    raft::device_matrix_view<ElementType, IndexType> const& A,
                                    raft::device_matrix_view<ElementType, IndexType> const& B,
                                    raft::device_matrix_view<ElementType, IndexType> C,
                                    ElementType w1, 
                                    ElementType w2) 
{
    IndexType n_row_a = A.extent(0);
    IndexType n_dim_a = A.extent(1);

    IndexType n_row_b = B.extent(0);
    IndexType n_dim_b = B.extent(1);

    assert(n_row_a == n_row_b && n_dim_a == n_dim_b);

    int block_dim_x = (n_row_a + block_size_x - 1) / block_size_x;
    int block_dim_y = (n_dim_a + block_size_y - 1) / block_size_y;

    dim3 full_blocks_per_grid(block_dim_x, block_dim_y);
    dim3 full_threads_per_block(block_size_x, block_size_y);

    matrix_weight_add_kernel<<<full_blocks_per_grid, full_threads_per_block>>>(A.data_handle(), B.data_handle(), C.data_handle(), 
        n_row_a, n_dim_a, w1, w2);
} 

template<typename ElementType, typename IndexType>
inline void refine(raft::device_resources const& dev_resources,
                   raft::device_matrix_view<ElementType, IndexType> const& dataset,
                   raft::device_matrix_view<ElementType, IndexType> const& queries,
                   raft::device_matrix_view<IndexType, IndexType> const& neighbor_candidates,
                   raft::device_matrix_view<IndexType, IndexType> indices, 
                   raft::device_matrix_view<ElementType, IndexType> distances)  
{
    IndexType n_data = dataset.extent(0);
    IndexType n_dim = dataset.extent(1);
    IndexType n_queries = queries.extent(0);

    IndexType n_candi = neighbor_candidates.extent(1);
    IndexType k = indices.extent(1);

    int full_blocks_per_grid_x = (n_queries + block_size_x - 1) / block_size_x;
    int full_blocks_per_grid_y = (n_candi + block_size_y - 1) / block_size_y;

    dim3 full_blocks_per_grid(full_blocks_per_grid_x, full_blocks_per_grid_y);
    dim3 full_threads_per_block(block_size_x, block_size_y);

    auto refine_dis = parafilter_mmr::make_device_matrix_view<ElementType, IndexType>(n_queries, n_candi);
    compute_l2_distances_kernel<<<full_blocks_per_grid, full_threads_per_block>>>(neighbor_candidates.data_handle(), 
        dataset.data_handle(), queries.data_handle(), refine_dis.data_handle(), n_data, n_queries, n_candi, n_dim);
    
    auto refine_indices = parafilter_mmr::make_device_matrix_view<IndexType, IndexType>(n_queries, k);
    raft::matrix::select_k<ElementType, IndexType>(dev_resources, refine_dis, std::nullopt, distances, refine_indices, true);

    select_elements<IndexType, IndexType>(dev_resources, neighbor_candidates, refine_indices, indices, false);
    LOG(INFO) << "cur query batch finished";
}

template<typename ElementType, typename IndexType>
void merge_intermediate_result(raft::device_resources const& dev_resources,
                               const std::string &file_path, 
                               IndexType batch_size, 
                               IndexType data_batch_size, 
                               IndexType n_queries, 
                               int topk, 
                               IndexType start_offset, 
                               int device_id,
                               raft::device_matrix_view<ElementType, IndexType> merged_dis_view, 
                               raft::device_matrix_view<IndexType, IndexType> merged_idx_view) 
{
    std::vector<std::vector<ElementType>> dis_matrices;
    std::vector<std::vector<IndexType>> idx_matrices;

    std::string dis_file_path = file_path + "distances_" + std::to_string(device_id);
    std::string neigh_file_path = file_path + "neighbors_" + std::to_string(device_id);

    read_matrices_from_file(dis_file_path, n_queries, topk, batch_size, dis_matrices);
    read_matrices_from_file(neigh_file_path, n_queries, topk, batch_size, idx_matrices);

    auto dis_view = parafilter_mmr::make_device_matrix_view<ElementType, IndexType>(n_queries, topk * batch_size);
    auto idx_view = parafilter_mmr::make_device_matrix_view<IndexType, IndexType>(n_queries, topk * batch_size);
    merge_matrices_to_gpu(dis_matrices, dis_view.data_handle(), n_queries, topk, batch_size);
    merge_matrices_to_gpu(idx_matrices, idx_view.data_handle(), n_queries, topk, batch_size);

    dim3 block_dim(block_size_x, block_size_y); 
    dim3 grid_dim((batch_size + block_dim.x - 1) / block_dim.x, (n_queries + block_dim.y - 1) / block_dim.y);
    modify_blocks<<<grid_dim, block_dim>>>(idx_view.data_handle(), n_queries, topk, 
                    batch_size, start_offset, data_batch_size);

    auto merged_idx_indirect_view =  parafilter_mmr::make_device_matrix_view<IndexType, IndexType>(n_queries, topk);
    
    raft::matrix::select_k<ElementType, IndexType>(
        dev_resources, dis_view, std::nullopt, merged_dis_view, merged_idx_indirect_view, true
    );

    select_elements<IndexType, IndexType>(dev_resources, idx_view, 
                         merged_idx_indirect_view, merged_idx_view, false);
}

template <typename m_t, typename idx_t = int>
void sliceMatrix(const m_t* in,
                 idx_t n_rows,
                 idx_t n_cols,
                 m_t* out,
                 idx_t x1,
                 idx_t y1,
                 idx_t x2,
                 idx_t y2,
                 bool row_major)
{
  auto lda = row_major ? n_cols : n_rows;
  dim3 block(256);
  dim3 grid(((x2 - x1) * (y2 - y1) + block.x - 1) / block.x);
  if (row_major)
    slice<<<grid, block>>>(in, lda, out, y1, x1, y2, x2);
  else
    slice<<<grid, block>>>(in, lda, out, x1, y1, x2, y2);
}

template <typename m_t, typename idx_t, typename layout_t>
void slice(raft::resources const& handle,
           raft::device_matrix_view<const m_t, idx_t, layout_t> in,
           raft::device_matrix_view<m_t, idx_t, layout_t> out,
           slice_coordinates<idx_t> coords)
{
  // todo: add parafilter expects to find the expected dimension semantics
  sliceMatrix(in.data_handle(),
                      in.extent(0),
                      in.extent(1),
                      out.data_handle(),
                      coords.row1,
                      coords.col1,
                      coords.row2,
                      coords.col2,
                      true);
}

void calc_pairwise_filter_dis(raft::device_matrix_view<float, uint64_t> const &data_labels, 
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

template<typename ElementType, typename IndexType>
struct cos_2_level_index_t {
    std::vector<raft::device_matrix_view<uint64_t, uint64_t>> clusters_list, 
    raft::device_matrix_view<uint64_t*, uint64_t> secondary_clusters_ptr,
    raft::device_vector_view<uint64_t, uint64_t> secondary_clusters_list_len,
    raft::device_matrix_view<float, uint64_t> centers, 
    raft::device_matrix_view<float, uint64_t> secondary_centers_list,
    raft::device_vector_view<int, uint64_t> inv_secondary_id_map; 
};

void group_by_cluster_id(raft::device_matrix_view<float, uint64_t> dataset, 
                         raft::device_vector_view<uint64_t, uint64_t> cluster_ids, 
                         std::vector<raft::device_matrix_view<uint64_t, uint64_t>>& grouped_data) 
{
    uint64_t n_data = dataset.extent(0);
    uint64_t n_dim = dataset.extent(1);

    int n_clusters = grouped_data.size();
    int* tmp_flags = parafilter::mem_allocator(n_data * sizeof(int));
    uint64_t* prefix_sum = parafilter::mem_allocator(n_data * sizeof(uint64_t));

    dim3 threads_block_size(256);
    dim3 blocks_per_grid(n_data + threads_block_size - 1 / threads_block_size);

    uint64_t tot_valid = 0;
    for (int i = 0; i < n_clusters; i++) {
        cudaMemset(tmp_flags, 0, n_data * sizeof(int));

        label_data_with_value_kernel<<<blocks_per_grid, threads_block_size>>>(tmp_flags, i, cluster_ids.data_handle(), n_data);

        thrust::device_ptr<int> thrust_flags(tmp_flags, tmp_flags + n_data);
        thrust::device_ptr<uint64_t> thrust_prefix_sum(prefix_sum, prefix_sum + n_data);
        thrust::inclusive_scan(thrust::device, thrust_flags, thrust_flags + n_data, thrust_prefix_sum);
        uint64_t n_valid = thrust::reduce(thrust::device, thrust_prefix_sum, thrust_prefix_sum + n_data, 0, thrust::maximum<uint64_t>());
        tot_valid += n_valid;
        
        auto cur_cluster_group = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_valid, 1);
        select_rows_by_flag_kernel<<<blocks_per_grid, threads_block_size>>>(tmp_flags, 
            prefix_sum, cur_cluster_group.data_handle(), n_data);
        grouped_data.push_back(cur_cluster_group);
    }
}

void select_valid_group(raft::device_matrix_view<uint64_t, uint64_t> const& selected_clusters, 
                        std::vector<raft::device_vector_view<uint64_t, uint64_t>> const& clusters_list, 
                        raft::device_matrix_view<float, uint64_t> const& ranges, 
                        raft::device_matrix_view<float, uint64_t> const& data_labels, 
                        int n_samples,
                        raft::device_matrix_view<int, uint64_t>& cnt_maxtix)
{
    uint64_t l = data_labels.extent(1);
    uint64_t n_data = data_labels.extent(0);
    uint64_t n_queries = selected_clusters.extent(0);
    uint64_t n_selected = selected_clusters.extent(1);
    uint64_t n_centers = cnt_maxtix.extent(1);

    std::vector<uint64_t> selected_cluster_host(n_queries * n_selected);
    cudaMemcpy(selected_cluster_host.data(), selected_clusters.data_handle(), 
        n_queries * n_selected * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    int* sampled_ids = parafilter_mmr::mem_allocator<int>(sizeof(int) * n_sample);

    int n_stream = 256;
    cudaStream_t streams[256];
    for (int i = 0; i < streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    for (uint64_t i = 0; i < n_queries; i++) {
        uint64_t *cur_id = selected_cluster_host.data() + i * n_selected;
        for (uint64_t j = 0; j < n_selected; j++) {
            uint64_t cluster_id = cur_id[j];
            
            auto cur_cluster_ids = clusters_list[cluster_id];
            uint64_t cur_cluster_len = cur_cluster_ids.extent(0);
            int cur_samples = std::min(n_samples, cur_cluster_len);

            dim3 gridSize(256);
            dim3 blockSize(cur_samples + full_threads_per_block.x - 1) / full_threads_per_block.x;

            curandState* d_states = parafilter_mmr::mem_allocator(cur_samples * sizeof(curandState));
            init_rng_kernel<<<gridSize, blockSize>>>(d_states, time(NULL));
            fill_random_kernel<<<gridSize, blockSize>>>(sampled_ids, d_states, cur_cluster_len, cur_samples);
            
            count_valid_elements_atomic_kernel<<<full_threads_per_block, full_blocks_per_grid, 0, streams[j % n_stream]>>>(
                ranges.data_handle() + 2 * l * i, 
                data_labels.data_handle(), cur_cluster_ids.data_handle(), l, n_data, cnt_maxtix.data_handle() + i * n_centers + cluster_id, 
                sampled_ids, n_samples);
        }
    }
}

void select_scondary_centers(raft::device_resources const& dev_resources,
                             raft::device_matrix_view<int, uint64_t> const& cnt_matrix, 
                             raft::device_matrix_view<float, uint64_t> const& secondary_centers,
                             raft::device_matrix_view<int, uint64_t> const& inv_secondary_id_map, 
                             raft::device_matrix_view<float, uint64_t> const& queries, 
                             int n_samples, 
                             raft::device_matrix_view<int, uint64_t> &selected_centers_id, 
                             raft::device_matrix_view<float, uint64_t> &selected_centers_dis)
{
    uint64_t n_queries = queries.extent(0);
    uint64_t n_dim = queries.extent(1);
    uint64_t n_secondary_centers = inv_secondary_id_map.extent(0);

    auto query_scondary_centers_dis = parafilter_mmr::make_device_matrix_view<float, int>(n_queries, n_secondary_centers);
    raft::distance::pairwise_distance(dev_resources, queries, secondary_centers, query_scondary_centers_dis);

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((n_queries + threads_per_block.x - 1) / threads_per_block.x, 
            (n_secondary_centers + threads_per_block.y - 1) / threads_per_block.y);

    select_valid_secondary_centers_dis_kernel<<<blocks_per_grid, threads_per_block>>>(cnt_matrix.data_handle(), 
        inv_secondary_id_map.data_handle(), query_scondary_centers_dis.data_handle(), n_samples, 0.5, n_queries, n_secondary_centers, n_dim);

    raft::matrix_select_k<float, uint64_t>(dev_resources, query_scondary_centers_dis, std::nullopt, selected_centers_dis, selected_centers_id, true);
}

template<typename ElementType, typename IndexType>
void parafilter_fill(ElementType* array,
                     ElementType value, 
                     IndexType* size)
{
    fill_kernel<<<(size + 255) / 256, 256>>>(array, value, size);
}

void merge_selected_indices(
    raft::device_matrix_view<uint64_t, uint64_t> const& selected_indices,
    raft::device_vector_view<uint64_t*, uint64_t> const& secondary_clusters_ptr,
    raft::device_vector_view<uint64_t, uint64_t> const& secondary_clusters_list_len,
    raft::device_matrix_view<uint64_t, uint64_t>& merged_indices)
{
    uint64_t n_queries = selected_indices.extent(0);
    uint64_t n_selected = selected_indices.extent(1);
    uint64_t row_stride = merged_indices.extent(1);  
    int block = 256;
    int grid = (n_queries + block - 1) / block;

    merge_selected_kernel<<<grid, block>>>(
        selected_indices.data_handle(),
        secondary_clusters_ptr.data_handle(),
        secondary_clusters_list_len.data_handle(),
        merged_indices.data_handle(),
        n_queries,
        n_selected,
        row_stride);
}