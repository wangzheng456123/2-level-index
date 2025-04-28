#pragma once
#include <stdint.h>
#include <raft/core/mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <core/mat_operators.cuh>
#include <utils/config_utils.h>
#include <utils/math_utils.h>
#include <utils/io_utils.cuh>

// A wrapper include all normalization
void preprocessing_labels(
        raft::device_matrix_view<float, uint64_t> data_labels, 
        raft::device_matrix_view<float, uint64_t> normalized_data_labels,
        raft::device_matrix_view<float, uint64_t> query_labels, 
        raft::device_matrix_view<float, uint64_t> normalized_query_labels,
        raft::device_matrix_view<float, uint64_t> denormalized_query_labels,
        float* global_min, 
        float* global_max,
        const filter_config& f_config, 
        bool is_query_changed = true,
        bool is_data_changed = true, 
        bool reconfig = false
);

void recovery_filters(
        raft::device_matrix_view<float, uint64_t> const& query_labels,
        raft::device_matrix_view<float, uint64_t> &ranges, 
        const filter_config& f_config
);

struct filter_parameters_dev_t {
        float* shift_val_dev = nullptr; 
        float* ranges_map_dev = nullptr; 
        float* shift_len_dev = nullptr;
        float* maps_len_dev = nullptr;
        int* div_value_dev = nullptr;

        int* map_types_dev = nullptr;
        bool is_configed = false;

        float* global_min_dev = nullptr;
        float* global_max_dev = nullptr;
};

void normalize_ranges(
        raft::device_matrix_view<float, uint64_t> ranges, 
        raft::device_matrix_view<float, uint64_t> normalized_ranges, 
        float* global_min_dev, 
        float* global_max_dev
);

void calculate_batch_min_max(raft::device_matrix_view<float, uint64_t> const&data_labels,
                             std::vector<float> &global_min, std::vector<float> &global_max,
                             uint64_t l);

void preprocessing_data_labels(raft::device_matrix_view<float, uint64_t> data_labels,   
        raft::device_matrix_view<float, uint64_t> normalized_data_labels, 
        float* global_max_dev);