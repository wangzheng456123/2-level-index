#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n";\
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

template <typename ElementType, typename IndexType>
__global__ void modify_blocks(ElementType* d_matrix, IndexType n_queries, IndexType topk, IndexType batch_size,
    ElementType start, IndexType data_batch_size) {
    IndexType row = blockIdx.x;                 // 每个块处理矩阵的一行
    IndexType block_idx = threadIdx.x;          // 每个线程处理一个 block
    IndexType block_size = topk;                // block 的大小

    if (row < n_queries && block_idx < batch_size) {
        IndexType block_start = row * (batch_size * block_size) + block_idx * block_size;
        ElementType add_value = start + block_idx * data_batch_size;
        for (IndexType i = 0; i < block_size; ++i) {
            d_matrix[block_start + i] += add_value;
        }
    }
}

int main() {
    try {
        // 参数
        using ElementType = float;   
        using IndexType = int;     

        const IndexType n_queries = 100;    // 矩阵的行数
        const IndexType topk = 10;          // 每个 block 的大小
        const IndexType batch_size = 5;     // 每行的 block 数
        const ElementType start = 1.0f;     // 起始值
        const IndexType data_batch_size = 2; // 增量步长
        const IndexType merged_cols = batch_size * topk;

        // 矩阵大小
        size_t matrix_size = n_queries * merged_cols;

        // 分配主机内存并初始化
        std::vector<ElementType> h_matrix(matrix_size, 0.0f);

        // 分配显存
        ElementType* d_matrix = nullptr;
        CUDA_CHECK(cudaMalloc(&d_matrix, matrix_size * sizeof(ElementType)));

        // 拷贝数据到显存
        CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix.data(), matrix_size * sizeof(ElementType), cudaMemcpyHostToDevice));

        // 定义 CUDA 配置
        dim3 grid_dim(n_queries);    // 每个块处理一行
        dim3 block_dim(batch_size);  // 每个线程处理一个 block

        // 启动 CUDA 核函数
        modify_blocks<ElementType, IndexType> << <grid_dim, block_dim >> > (d_matrix, n_queries, topk, batch_size, start, data_batch_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 将结果拷回主机
        CUDA_CHECK(cudaMemcpy(h_matrix.data(), d_matrix, matrix_size * sizeof(ElementType), cudaMemcpyDeviceToHost));

        // 打印部分结果验证
        std::cout << "Modified matrix (partial output):" << std::endl;
        for (IndexType row = 0; row < std::min(n_queries, 5); ++row) {
            for (IndexType col = 0; col < merged_cols; ++col) {
                std::cout << h_matrix[row * merged_cols + col] << " ";
            }
            std::cout << std::endl;
        }

        // 释放显存
        CUDA_CHECK(cudaFree(d_matrix));

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
