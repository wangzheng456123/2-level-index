#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

template<typename ElmentType, typename IndexType>
__global__ void min_max_reduce_kernel(ElmentType* input, ElmentType* output, IndexType n, bool is_min = true)
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
                input[left] = min(input[right], input[left]);
            else input[left] = max(input[right], input[left]);
        }
        __syncthreads();
    }

    if (!thread_id)
    {
        output[block_id] = input[block_start];
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

// 测试用例
void test_array_min_max_reduce() {
    using ElementType = float;
    using IndexType = int;

    // 测试数据
    std::vector<ElementType> h_array(256 * 10 + 79);
    for (auto& data : h_array) {
        data = rand() % 114514;
    }
    IndexType n_elements = h_array.size();

    // 分配 GPU 内存
    ElementType* d_array;
    cudaMalloc(&d_array, n_elements * sizeof(ElementType));

    // 将数据从主机复制到设备
    cudaMemcpy(d_array, h_array.data(), n_elements * sizeof(ElementType), cudaMemcpyHostToDevice);

    // 使用 array_min_max_reduce 函数计算最小值
    ElementType gpu_min = array_min_max_reduce<ElementType, IndexType>(d_array, n_elements, true);

    // 使用 array_min_max_reduce 函数计算最大值
    ElementType gpu_max = array_min_max_reduce<ElementType, IndexType>(d_array, n_elements, false);

    // 在 CPU 上计算参考结果
    ElementType cpu_min = *std::min_element(h_array.begin(), h_array.end());
    ElementType cpu_max = *std::max_element(h_array.begin(), h_array.end());

    // 比较结果
    std::cout << "CPU Min: " << cpu_min << ", GPU Min: " << gpu_min << std::endl;
    std::cout << "CPU Max: " << cpu_max << ", GPU Max: " << gpu_max << std::endl;

    // 验证结果
    if (std::abs(cpu_min - gpu_min) < 1e-5f && std::abs(cpu_max - gpu_max) < 1e-5f) {
        std::cout << "Test Passed!" << std::endl;
    }
    else {
        std::cout << "Test Failed!" << std::endl;
    }

    // 释放 GPU 内存
    cudaFree(d_array);
}

int main() {
    test_array_min_max_reduce();
    return 0;
}
