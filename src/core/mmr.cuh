#pragma once
#include <raft/core/device_mdarray.hpp>
#include <map>
#include <stdint.h>
#include <vector>
#include <utils/debugging_utils.cuh>
#include "cuda_runtime.h"
#include "easylogging++.h"

#define INIT_PARAFILTER_MMR_STATIC_MEMBERS                         \
    std::vector<uint64_t> parafilter_mmr::total;                   \
    std::vector<uint64_t> parafilter_mmr::available;               \
    std::vector<std::map<uint64_t, std::vector<void *>>> parafilter_mmr::cur_mems; \
    std::vector<std::map<uint64_t, int>> parafilter_mmr::cur_offset;

class parafilter_mmr {
public:
    static void init_mmr();
  
    static uint64_t get_current_workspace_free() 
    {
      int id;
      cudaGetDevice(&id);
      return available[id];
    }   
  
    static uint64_t get_current_workspace_total() 
    {
      int id;
      cudaGetDevice(&id);
      return total[id];
    }

    static uint64_t get_current_workspace_used() {
      int id;
      cudaGetDevice(&id);
      return total[id] - available[id];
    }
  
    static void reset_current_workspace(uint64_t size) {
      int id;
      cudaGetDevice(&id);
      total[id] = size; 
      available[id] = size; 
    }
  
    template<typename ElementType, typename IndexType>
    static auto make_device_matrix_view(IndexType n_row, 
                                        IndexType n_dim) 
    {
      ElementType *device_ptr;
      uint64_t size = sizeof(ElementType) * n_row * n_dim;
      device_ptr = (ElementType*)mem_allocator(size);
      LOG(INFO) << "alloc memory with size: " << size;
      return raft::make_device_matrix_view<ElementType, IndexType>(device_ptr, n_row, n_dim);
    }
    
    template<typename ElementType, typename IndexType>
    static auto make_device_vector_view(IndexType n_elements) {
      ElementType *device_ptr;
      uint64_t size = sizeof(ElementType) * n_elements;
      device_ptr = (ElementType*)mem_allocator(size);
      LOG(INFO) << "alloc memory with size: " << size;
      return raft::make_device_vector_view<ElementType, IndexType>(device_ptr, n_elements);
    }
  
    static void workspace_add_mem(void *mem, uint64_t size) {
      int id;
      cudaGetDevice(&id);
      available[id] -= size;
      LOG(INFO) << "add device mem with" << mem;
      cur_mems[id][size].push_back(mem);
    }
  
    static void* mem_allocator(uint64_t size);
  
    static void free_cur_workspace_device_mems(bool free_mems = true); 
  
    // todo: make these data thread safe
    static std::vector<uint64_t> total;
    static std::vector<uint64_t> available;
    // todo: implement mmr deconstructor to avoid GPU mem leak 
    static std::vector<std::map<uint64_t, std::vector<void *>>> cur_mems;
    static std::vector<std::map<uint64_t, int>> cur_offset;
     ~parafilter_mmr();
private:
};