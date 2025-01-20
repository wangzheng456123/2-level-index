#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include "para_filter.cuh"

// Host function to launch the kernel
template <typename CodebookType, typename ElementType, typename IndexType>
void compute_result_batched_cuda(
    const CodebookType* d_codebook,   // Device pointer: codebook
    const ElementType* d_lut,         // Device pointer: LUT
    ElementType* d_result,            // Device pointer: result
    IndexType n_data,                 // Number of data points
    IndexType pq_dim,                 // Number of dimensions
    IndexType n_clusters,             // Number of clusters
    IndexType n_queries,              // Number of queries
    IndexType data_batch_size,        // Batch size for data
    IndexType query_batch_size)       // Batch size for queries
{
    // Determine grid and block sizes
    dim3 block_size(16, 16);  // Threads per block
    dim3 grid_size((n_data + block_size.x * data_batch_size - 1) / (block_size.x * data_batch_size),
        (n_queries + block_size.y * query_batch_size - 1) / (block_size.y * query_batch_size));

    // Launch the kernel
    compute_batched_L2_distance_kernel<CodebookType, ElementType, IndexType>
        << <grid_size, block_size >> > (
            d_codebook, d_lut, d_result, n_data, pq_dim, n_clusters, n_queries,
            data_batch_size, query_batch_size);

    cudaDeviceSynchronize();
}

int main() {
    using CodebookType = uint8_t;
    using ElementType = float;
    using IndexType = int;

    // Parameters
    constexpr IndexType n_data = 4;
    constexpr IndexType pq_dim = 2;
    constexpr IndexType n_clusters = 3;
    constexpr IndexType n_queries = 3;
    constexpr IndexType data_batch_size = 2;
    constexpr IndexType query_batch_size = 2;

    // Initialize host data
    CodebookType h_codebook[pq_dim * n_data] = {
        0, 2, 1, 2, 
        1, 0, 2, 1
    };

    ElementType h_lut[n_queries * pq_dim * n_clusters] = {
        // Query 1
        0.1f, 0.2f, 0.3f,  // Dimension 1, clusters 0-2
        0.4f, 0.5f, 0.6f,  // Dimension 2, clusters 0-2

        // Query 2
        0.2f, 0.3f, 0.4f,  // Dimension 1, clusters 0-2
        0.5f, 0.6f, 0.7f,  // Dimension 2, clusters 0-2

        // Query 3
        0.3f, 0.4f, 0.5f,  // Dimension 1, clusters 0-2
        0.6f, 0.7f, 0.8f   // Dimension 2, clusters 0-2
    };

    ElementType h_result[n_queries * n_data] = { 0 };  // Initialize result to zero

    // Allocate device memory
    CodebookType* d_codebook;
    ElementType* d_lut;
    ElementType* d_result;

    cudaMalloc(&d_codebook, sizeof(h_codebook));
    cudaMalloc(&d_lut, sizeof(h_lut));
    cudaMalloc(&d_result, sizeof(h_result));

    cudaMemcpy(d_codebook, h_codebook, sizeof(h_codebook), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lut, h_lut, sizeof(h_lut), cudaMemcpyHostToDevice);

    // Call the host function to launch the kernel
    compute_result_batched_cuda(d_codebook, d_lut, d_result,
        n_data, pq_dim, n_clusters, n_queries,
        data_batch_size, query_batch_size);

    // Copy the result back to host
    cudaMemcpy(h_result, d_result, sizeof(h_result), cudaMemcpyDeviceToHost);

    // Expected results (pre-computed manually):
    // Result matrix (queries x data):
    // Query 1: [0.6, 0.5, 0.9, 0.8]
    // Query 2: [0.8, 0.7, 1.1, 1.0]
    // Query 3: [1.0, 0.9, 1.3, 1.2]
    ElementType expected_result[n_queries * n_data] = {
        0.6f, 0.7f, 0.8f, 0.8f,  // Query 1
        0.8f, 0.9f, 1.0f, 1.0f,  // Query 2
        1.0f, 1.1f, 1.2f, 1.2f   // Query 3
    };

    // Verify results
    for (IndexType q = 0; q < n_queries; q++) {
        for (IndexType d = 0; d < n_data; d++) {
            ElementType computed = h_result[q * n_data + d];
            ElementType expected = expected_result[q * n_data + d];

            assert(fabs(computed - expected) < 1e-5);
            std::cout << "Query " << q << ", Data " << d
                << ": Computed = " << computed
                << ", Expected = " << expected << "\n";
        }
    }

    std::cout << "\nAll results are correct!\n";

    // Free device memory
    cudaFree(d_codebook);
    cudaFree(d_lut);
    cudaFree(d_result);

    return 0;
}

