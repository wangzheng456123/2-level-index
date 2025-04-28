#pragma once
#include <core/preprocessing.cuh>
#include <core/filter.cuh>
#include <index/product-quantization.cuh>

namespace prefiltering {
    struct index_data_t {
        raft::device_matrix_view<float, uint64_t> data_labels;
    };

    class index : public pq::index {
    public: 
        index(pq::index_params const& pq_config, std::string const& filter_config_path, uint64_t topk) : 
                pq::index(pq_config, topk), f_config(filter_config_path) {}
        void build(build_input_t const& in) override;
        void query(query_input_t const& in, query_output_t &out) const override;
    protected:
        void query_prefilter(query_input_t const& in, valid_indices_t const& valid_indices_in, query_output_t &out) const;
        void build_filters_and_labels(build_input_t const& in);
        void prefilter(query_input_t const& in, valid_indices_t const& valid_indices_in, valid_indices_t &valid_indices_out) const;
        filter_config f_config;
        index_data_t filters_and_labels;
    private:
    }
;}