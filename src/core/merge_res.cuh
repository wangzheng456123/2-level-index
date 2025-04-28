#pragma once
#include <string>
#include <stdint.h>
#include <vector>
#include <core/mmr.cuh>
#include <core/mat_operators.cuh>
#include <raft/core/device_resources.hpp>
#include <raft/matrix/select_k.cuh>
#include <raft/core/device_mdspan.hpp>
#include <utils/io_utils.cuh>

void merge_intermediate_result(
    raft::device_resources const& dev_resources,
    const std::string &file_path, 
    uint64_t batch_size, 
    uint64_t data_batch_size, 
    uint64_t n_queries, 
    int topk, 
    uint64_t start_offset, 
    int device_id,
    raft::device_matrix_view<float, uint64_t> merged_dis_view, 
    raft::device_matrix_view<uint64_t, uint64_t> merged_idx_view);