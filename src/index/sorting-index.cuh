#pragma once
#include <index/prefiltering.cuh>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

namespace sorting {
    struct index_data_t {
        raft::device_matrix_view<float, uint64_t> values_view;
        raft::device_matrix_view<uint64_t, uint64_t> indices_view;
    };

    class index : public prefiltering::index {
    public: 
        index(pq::index_params const& pq_config, std::string const& filter_config_path, uint64_t topk) : 
            prefiltering::index(pq_config, filter_config_path, topk) {}
        void build(build_input_t const& in) override;
        void query(query_input_t const& in, query_output_t &out) const override;
    private: 
        index_data_t sorting_index;
        void sorting_data_labels(build_input_t const&in);
        void query_sorting_indices(query_input_t const &in, valid_indices_t &valid_indices) const;
    };

}