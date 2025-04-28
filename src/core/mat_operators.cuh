#pragma once
#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdspan.hpp>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <curand_kernel.h>
#include <utils/debugging_utils.cuh>

const int block_size = 128;
const int block_size_x = 32;
const int block_size_y = 16;

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

template <typename T>
__global__ void fill_kernel(T* data, T value, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = value;
}

template <typename T>
__global__ void init_rng_kernel(curandState* states, T seed) {
    T idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

template <typename T>
__global__ void fill_random_kernel(T* out, curandState* states, T range, T size) {
    T idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int r = curand(&states[idx]) % range;
        out[idx] = r;
    }
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

template<typename ElementType, typename IndexType>
__global__ void shuffle_data_kernel(const ElementType* in, ElementType* out,
                                    IndexType n_queries, IndexType l)
{
    IndexType row = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n_queries && col < l) {
        ElementType lo = in[row * 2 * l + 2 * col];
        ElementType hi = in[row * 2 * l + 2 * col + 1];

        out[col * n_queries + row] = lo;
        out[(col + l) * n_queries + row] = hi;
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

template<typename ElementType, typename IndexType>
void shuffle_data(raft::device_matrix_view<ElementType, IndexType> in_mat, 
                  raft::device_matrix_view<ElementType, IndexType> out_mat) 
{
    IndexType n_queries = in_mat.extent(0);
    IndexType l = in_mat.extent(1) / 2;

    dim3 block_dim(32, 8); 
    dim3 grid_dim((n_queries + block_dim.x - 1) / block_dim.x,
                  (l + block_dim.y - 1) / block_dim.y);

    shuffle_data_kernel<<<grid_dim, block_dim>>>(
        in_mat.data_handle(),
        out_mat.data_handle(),
        n_queries,
        l
    );
}

template <typename InputType, typename OutputType, typename IndexType>
void matrix_scan(raft::device_matrix_view<InputType, IndexType> const &in_matrix,
                 raft::device_matrix_view<OutputType, IndexType> &out_matrix) {
    IndexType rows = in_matrix.extent(0);
    IndexType cols = in_matrix.extent(1);

    // Ensure in_matrix and out_matrix have the same shape
    assert(in_matrix.extent(0) == out_matrix.extent(0));
    assert(in_matrix.extent(1) == out_matrix.extent(1));

    // Launch parallel scan for each row
    for (IndexType row = 0; row < rows; ++row) {
        InputType *in_row_ptr = in_matrix.data_handle() + row * cols;
        OutputType *out_row_ptr = out_matrix.data_handle() + row * cols;

        thrust::inclusive_scan(thrust::device, in_row_ptr, in_row_ptr + cols, out_row_ptr);
    }
}

template<typename ElementType, typename IndexType>
void fill(ElementType* array,
          ElementType value, 
          IndexType size)
{
    fill_kernel<<<(size + 255) / 256, 256>>>(array, value, size);
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
           raft::device_matrix_view<m_t, idx_t, layout_t> const& in,
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

// select elements of in an indices set from the input matrix and put it into the output matrix
template<typename ElementType, typename IndexType>
inline void select_elements(raft::device_resources const& dev_resources,
                            raft::device_matrix_view<ElementType, IndexType> const& input_matrix,
                            raft::device_matrix_view<IndexType, IndexType> const& indices, 
                            raft::device_matrix_view<ElementType, IndexType> &output,
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
    checkCUDAErrorWithLine("select elements kernel failde:");
}

// ond dimensional version wrapper for matrix select elements
template<typename ElementType, typename IndexType>
inline void select_elements(raft::device_resources const& dev_resources,
                            raft::device_vector_view<ElementType, IndexType> const& input_vector,
                            raft::device_vector_view<IndexType, IndexType> const& indices_vector, 
                            raft::device_vector_view<ElementType, IndexType> &output_vector
                            ) 
{
    auto input_matrix = raft::make_device_matrix_view<ElementType, IndexType>(input_vector.data_handle(), 1, input_vector.extent(0));
    auto indices = raft::make_device_matrix_view<ElementType, IndexType>(indices_vector.data_handle(), 1, indices_vector.extent(0));
    auto output = raft::make_device_matrix_view<ElementType, IndexType>(output_vector.data_handle(), 1, output_vector.extent(0));

    select_elements(dev_resources, input_matrix, indices, output, false);
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
