#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// select a batch of idices of row from the input matrix
template<typename ElementType, typename IndexType>
__global__ void select_elements_kernel(const ElementType* input, const IndexType* indices,
    ElementType* output, int n_row, int n_dim_o, int n_dim_i, bool is_select_row)
{
    IndexType x = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= n_row || y >= n_dim_o) return;
    IndexType idx = x;
    IndexType idy = indices[x * n_dim_o + y];

    if (is_select_row) {
        for (IndexType i = 0; i < n_dim_i; i++) {
            IndexType o_idx = x * n_dim_i * n_dim_o + y * n_dim_i + i;
            output[o_idx] = input[n_dim_i * idy + i];
        }
    }
    else {
        IndexType o_idx = x * n_dim_o + y;
        output[o_idx] = input[n_dim_i * idx + idy];
    }
}

// Host code to test the kernel
void test_select_elements_kernel() {
    // Parameters
    const int n_row = 2;
    const int n_dim_o = 3;
    const int n_dim_i = 4;
    bool is_select_row;

    // Input data
    std::vector<float> input = {
        1, 2, 3, 4,    // row 0
        5, 6, 7, 8,    // row 1
        9, 10, 11, 12  // row 2
    };

    std::vector<int> indices = {
        2, 0, 1,  // row 0
        1, 2, 0   // row 1
    };

    // Host output buffers
    std::vector<float> host_output;

    // Device memory
    float* d_input, * d_output;
    int* d_indices;

    // Allocate device memory
    size_t input_size = input.size() * sizeof(float);
    size_t indices_size = indices.size() * sizeof(int);
    size_t output_size;

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_indices, indices_size);

    // Copy input data to device
    cudaMemcpy(d_input, input.data(), input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices.data(), indices_size, cudaMemcpyHostToDevice);

    // Test case 1: is_select_row = true
    is_select_row = true;
    output_size = n_row * n_dim_o * n_dim_i * sizeof(float);
    host_output.resize(n_row * n_dim_o * n_dim_i);
    cudaMalloc(&d_output, output_size);

    dim3 block_dim(16, 16);
    dim3 grid_dim((n_row + block_dim.x - 1) / block_dim.x, (n_dim_o + block_dim.y - 1) / block_dim.y);

    select_elements_kernel << <grid_dim, block_dim >> > (d_input, d_indices, d_output, n_row, n_dim_o, n_dim_i, is_select_row);
    cudaMemcpy(host_output.data(), d_output, output_size, cudaMemcpyDeviceToHost);

    // Verify the result
    std::vector<float> expected_output_1 = {
        9, 10, 11, 12,  1, 2, 3, 4,  5, 6, 7, 8,
        5, 6, 7, 8,  9, 10, 11, 12,  1, 2, 3, 4
    };

    std::cout << "Test case 1 (is_select_row = true): \n";
    for (size_t i = 0; i < host_output.size(); ++i) {
        std::cout << host_output[i] << " ";
        if ((i + 1) % n_dim_i == 0) std::cout << "| ";
    }
    std::cout << "\n\n";

    // Test case 2: is_select_row = false
    is_select_row = false;
    output_size = n_row * n_dim_o * sizeof(float);
    host_output.resize(n_row * n_dim_o);
    cudaMalloc(&d_output, output_size);

    select_elements_kernel << <grid_dim, block_dim >> > (d_input, d_indices, d_output, n_row, n_dim_o, n_dim_i, is_select_row);
    cudaMemcpy(host_output.data(), d_output, output_size, cudaMemcpyDeviceToHost);

    // Verify the result
    std::vector<float> expected_output_2 = {
        3, 1, 2,   // row 0
        6, 7, 5    // row 1
    };

    std::cout << "Test case 2 (is_select_row = false): \n";
    for (size_t i = 0; i < host_output.size(); ++i) {
        std::cout << host_output[i] << " ";
        if ((i + 1) % n_dim_o == 0) std::cout << "| ";
    }
    std::cout << "\n";

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_indices);
    cudaFree(d_output);
}

int main() {
    test_select_elements_kernel();
    return 0;
}
