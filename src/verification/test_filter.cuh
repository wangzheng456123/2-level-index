#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <cassert>
#include "para_filter.cuh"

// 测试用例
void test_filter_by_constrain_kernel() {
    // 输入数据
    const int n_queries = 2;
    const int n_candi = 3;
    const int l = 2;

    // 初始化输入矩阵
    std::vector<float> h_dis = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    std::vector<float> h_constrains = {
        1.0f, 4.0f,   // 第一个查询的约束
        2.0f, 5.0f,
        0.0f, 3.0f,   // 第二个查询的约束
        1.0f, 2.0f
    };
    std::vector<float> h_candi_label = {
        1.5f, 4.5f, 0.5f, 2.5f, 3.0f, 4.0f,   
        2.5f, 3.5f, 1.5f, 5.5f, 1.0f, 3.5f
    };

    // 初始化设备内存
    float* d_dis, * d_constrains, * d_candi_label;
    cudaMalloc(&d_dis, n_queries * n_candi * sizeof(float));
    cudaMalloc(&d_constrains, n_queries * l * 2 * sizeof(float));
    cudaMalloc(&d_candi_label, n_queries * l * n_candi * sizeof(float));

    // 拷贝数据到设备
    cudaMemcpy(d_dis, h_dis.data(), n_queries * n_candi * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_constrains, h_constrains.data(), n_queries * l * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_candi_label, h_candi_label.data(), n_queries * l * n_candi * sizeof(float), cudaMemcpyHostToDevice);

    // 配置 CUDA 网格和线程块
    dim3 block(16, 16);
    dim3 grid((n_queries + block.x - 1) / block.x, (n_candi + block.y - 1) / block.y);

    // 启动 CUDA 核函数
    filter_by_constrain_kernel << <grid, block >> > (d_dis, d_constrains, d_candi_label, l, n_queries, n_candi);

    // 同步设备
    cudaDeviceSynchronize();

    // 拷贝结果回主机
    std::vector<float> h_dis_result(n_queries * n_candi);
    cudaMemcpy(h_dis_result.data(), d_dis, n_queries * n_candi * sizeof(float), cudaMemcpyDeviceToHost);

    // 计算期望结果
    std::vector<float> expected_dis = {
        1.0f, 0.0f, 3.0f,  // 第一个查询
        0.0f, 0.0f, 0.0f   // 第二个查询
    };

    // 验证结果
    bool passed = true;
    for (int i = 0; i < n_queries * n_candi; i++) {
        if (std::abs(h_dis_result[i] - expected_dis[i]) > 1e-5f) {
            passed = false;
            break;
        }
    }

    // 打印结果
    if (passed) {
        std::cout << "Test Passed!" << std::endl;
    }
    else {
        std::cout << "Test Failed!" << std::endl;
        std::cout << "Expected: ";
        for (const auto& val : expected_dis) std::cout << val << " ";
        std::cout << "\nGot: ";
        for (const auto& val : h_dis_result) std::cout << val << " ";
        std::cout << std::endl;
    }

    // 释放设备内存
    cudaFree(d_dis);
    cudaFree(d_constrains);
    cudaFree(d_candi_label);
}

int main() {
    test_filter_by_constrain_kernel();
    return 0;
}
