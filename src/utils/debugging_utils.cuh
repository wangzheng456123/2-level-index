#pragma once
#include <cuda_runtime.h>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdspan.hpp>
#include "easylogging++.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
/**
* Check for CUDA errors; print and exit if there was a problem.
*/
inline void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      LOG(ERROR) << "Line :" << line;
    }
    LOG(ERROR) << "Cuda error :" << msg << ":" << cudaGetErrorString(err);
    exit(EXIT_FAILURE);
  }
}

template<typename T, typename ElementType>
void print_raft_view(raft::device_resources const &dev_resources, 
                    T const& device_view) 
{
  size_t n_row = device_view.extent(0);
  size_t n_dim = device_view.extent(1);

  std::cout << "matrix view with rows: " << n_row << ", dimension: " << n_dim << "\n";

  // for 1 dimensional data, extent 1 is 0
  if (!n_dim) n_dim = 1;
  ElementType* host_ptr = new ElementType[n_row * n_dim];

  cudaMemcpy(host_ptr, device_view.data_handle(), n_row * n_dim * sizeof(ElementType), cudaMemcpyDeviceToHost);
  std::cout << "copy success!\n";

  for (int i = 0; i < n_row; i++) {
    std::cout << "[";
    for (int j = 0; j < n_dim; j++) {
      std::cout << host_ptr[i * n_dim + j];
      if (j != n_dim - 1)
        std::cout << ",";
    }
    std::cout << "]\n";
  }

  delete [] host_ptr;
}

template<typename T> void dbg_print_cuda_device_mem(T* p, int64_t size) 
{
    int n_ele = size / sizeof(T);
    T* host_ptr = new T[n_ele];

    cudaMemcpy(host_ptr, p, size, cudaMemcpyDeviceToHost);
    checkCUDAErrorWithLine("cuda error with cuda memcpy failed!");

    for (int i = 0; i < n_ele; i++)
        std::cout << host_ptr[i] << "\n";
}