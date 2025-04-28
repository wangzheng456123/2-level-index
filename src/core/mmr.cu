#include <core/mmr.cuh>

void parafilter_mmr::init_mmr() {
    int device_count;
    cudaGetDeviceCount(&device_count);

    total.resize(device_count);
    available.resize(device_count);
    cur_mems.resize(device_count);
    cur_offset.resize(device_count);
}

void* parafilter_mmr::mem_allocator(uint64_t size) 
{
    int id;
    cudaGetDevice(&id);
    LOG(INFO) << "allocate memory with " << size << " byte on device :" << id; 
    if (id < cur_mems.size() && cur_mems[id].count(size) && cur_offset[id][size] < cur_mems[id][size].size()) {
      int offset = cur_offset[id][size];
      cur_offset[id][size]++;
      LOG(INFO) << "parafilter mmr allocate block from pool";
      return cur_mems[id][size][offset];
    }
    else {
      void* mem;
      cudaMalloc((void**)&mem, size);
      LOG(INFO) << "parafilter mmr allocate block runtime";
      checkCUDAErrorWithLine("cudaMalloc failed");
      workspace_add_mem(mem, size);
      cur_offset[id][size]++;
      LOG(INFO) << cur_mems[id][size].size() << " blocks, " << cur_offset[id][size]
        << " block offset";
      return mem;
    }
}

void parafilter_mmr::free_cur_workspace_device_mems(bool free_mems) 
{
    int id;
    cudaGetDevice(&id);
    for (auto iter = cur_offset[id].begin(); iter != cur_offset[id].end(); ++iter) {
      iter->second = 0;
    }
    if (free_mems) {
      for (auto iter = cur_mems[id].begin(); iter != cur_mems[id].end(); ++iter) {
        auto key = iter->first;
        for (auto _iter = cur_mems[id][key].begin(); _iter != cur_mems[id][key].end(); ++_iter) { 
            LOG(INFO) << "free device mem with" << *_iter;
            cudaFree(*_iter);
            checkCUDAErrorWithLine("free work space memory failed");
        }
        cur_mems[id][key].clear();
      }
      cur_mems[id].clear();
    }
}




