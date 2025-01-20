#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include "para_filter.cuh"

// CUDA kernel template

// Helper function for device memory allocation
template <typename T>
T* allocate_device_memory(const std::vector<T>& host_data) {
    T* device_ptr;
    cudaMalloc(&device_ptr, host_data.size() * sizeof(T));
    cudaMemcpy(device_ptr, host_data.data(), host_data.size() * sizeof(T), cudaMemcpyHostToDevice);
    return device_ptr;
}

// Helper function to copy device memory to host
template <typename T>
std::vector<T> copy_device_to_host(T* device_ptr, size_t size) {
    std::vector<T> host_data(size);
    cudaMemcpy(host_data.data(), device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
    return host_data;
}

// Main test function
void test_build_pq_lut_kernel_correctness() {
    using ElementType = float;
    using IndexType = int;

    // Test configuration
    IndexType n_queries = 2;
    IndexType pq_dim = 2;
    IndexType pq_len = 2;
    IndexType n_clusters = 2;
    IndexType n_dim = pq_dim * pq_len;
    IndexType query_batch_size = 1;

    // Input data
    std::vector<ElementType> centers = {
        // pq_dim: 0
        1.0f, 2.0f, 
        3.0f, 4.0f,
        // pq_dim: 1
        5.0f, 6.0f, 
        7.0f, 8.0f
    }; // Shape: [pq_dim * pq_len * n_clusters]

    std::vector<ElementType> queries = {
        1.5f, 2.5f, 3.5f, 4.5f,
        5.5f, 6.5f, 7.5f, 8.5f
    }; // Shape: [n_queries * n_dim]

    std::vector<ElementType> expected_lut = {
        0.5f, 4.5f,
        4.5f, 24.5f,
        40.5f, 12.5f,
        12.5f, 0.5f
    }; // Shape: [n_queries * pq_dim * n_clusters]

    // Allocate device memory
    ElementType* d_centers = allocate_device_memory(centers);
    ElementType* d_queries = allocate_device_memory(queries);
    ElementType* d_lut;
    cudaMalloc(&d_lut, n_queries * pq_dim * n_clusters * sizeof(ElementType));

    // Launch kernel
    dim3 threads_per_block(8, 8, 2);
    dim3 num_blocks(((n_queries + query_batch_size - 1) / query_batch_size + 7) / 8, (n_clusters + 7) / 8, (pq_dim + 1) / 2);

    build_pq_lut_kernel<ElementType, IndexType> << <num_blocks, threads_per_block >> > (
        d_centers, d_queries, query_batch_size, d_lut,
        pq_len, pq_dim, n_dim, n_queries, n_clusters
        );

    // Copy result back to host
    std::vector<ElementType> result = copy_device_to_host(d_lut, n_queries * pq_dim * n_clusters);

    // Verify results
    for (size_t i = 0; i < expected_lut.size(); i++) {
        if (std::fabs(result[i] - expected_lut[i]) > 1e-6) {
            std::cerr << "Test failed at index " << i << ": expected "
                << expected_lut[i] << ", got " << result[i] << std::endl;
            cudaFree(d_centers);
            cudaFree(d_queries);
            cudaFree(d_lut);
            return;
        }
    }

    std::cout << "Test passed!" << std::endl;

    // Free device memory
    cudaFree(d_centers);
    cudaFree(d_queries);
    cudaFree(d_lut);
}


int main() {
    test_build_pq_lut_kernel_correctness();
    return 0;
}
