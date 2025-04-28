#pragma once
#include <core/mat_operators.cuh>
#include <core/mmr.cuh>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/matrix/select_k.cuh>

void refine(
    raft::device_resources const& dev_resources,
    raft::device_matrix_view<float, uint64_t> const& dataset,
    raft::device_matrix_view<float, uint64_t> const& queries,
    raft::device_matrix_view<uint64_t, uint64_t> const& neighbor_candidates,
    raft::device_matrix_view<uint64_t, uint64_t> &indices, 
    raft::device_matrix_view<float, uint64_t> &distances
);