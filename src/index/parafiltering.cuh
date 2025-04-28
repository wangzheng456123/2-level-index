#pragma once
#include <index/product-quantization.cuh>
#include <core/preprocessing.cuh>
#include <core/filter.cuh>

namespace parafiltering {
    struct index_data_t {
        raft::device_matrix_view<float, uint64_t> data_labels;
    };

    struct index_params {
        uint64_t exps1;
        float merge_rate;
    };

    struct config_t: pq::config_t {
        // parameters for build pq index
        index_params parafilter_config;

        config_t(std::string const& path):
            pq::config_t(path, [](std::map<std::string, int> &m) {
               m.insert({ 
                            {"MERGE_RATE", OFFSETOF_NESTED(config_t, parafilter_config, index_params, merge_rate)},
                            {"EXPS1", OFFSETOF_NESTED(config_t, parafilter_config, index_params, exps1)}
                        }); 
            }) {  }
    };

    class index : public pq::index {
    public:
        index(pq::index_params const& pq_params, index_params const& parafilter_params, std::string const& filter_config_path, uint64_t topk)
            : pq::index(pq_params, topk), parafilter_config(parafilter_params), f_config(filter_config_path) {}
        void build(build_input_t const&) override;
        void query(query_input_t const&, query_output_t &) const override;
    private:
        filter_config f_config;
        index_data_t normalized_labels;
        index_params parafilter_config;
    };

}