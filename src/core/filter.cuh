#pragma once
#include <stdint.h>
#include <raft/core/device_resources.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/matrix/select_k.cuh>
#include <core/mat_operators.cuh>
#include <core/mmr.cuh>

void filter_candi_by_labels(
    raft::device_resources const& dev_resources,
    raft::device_matrix_view<float, uint64_t> const& candi_labels, 
    raft::device_matrix_view<float, uint64_t> const& constrains, 
    raft::device_matrix_view<float, uint64_t> const& pq_dis,
    int topk, 
    raft::device_matrix_view<uint64_t, uint64_t> fcandi);

uint64_t filter_valid_data(raft::device_resources const& dev_resources,
    raft::device_matrix_view<float, uint64_t> const &data_labels, 
    raft::device_matrix_view<float, uint64_t> const &constrains,
    raft::device_matrix_view<uint64_t, uint64_t> &valid_indices, 
    raft::device_matrix_view<uint64_t, uint64_t> const &coarse_filtered_indices = 
                raft::device_matrix_view<uint64_t, uint64_t>{}, 
    bool is_filter = false
);

void label_matrix_by_filter(
    raft::device_matrix_view<float, uint64_t> const &labels, 
    raft::device_matrix_view<float, uint64_t> const &ranges, 
    raft::device_matrix_view<float, uint64_t> &matrix); 