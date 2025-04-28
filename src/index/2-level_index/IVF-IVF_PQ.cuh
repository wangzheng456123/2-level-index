#include <index/2-level_index/X-IVF_PQ.cuh>

namespace IVF_IVF_PQ {
    struct index_data_t {
        raft::device_matrix_view<float, uint64_t> centers;
    };
    struct index_params {
        uint64_t clusters0;
    };
    struct config_t: X_IVF_PQ::config_t {
        index_params ivf_config;

        config_t(std::string const& path):
            X_IVF_PQ::config_t(path, [](std::map<std::string, int> &m) {
               m.insert({ 
                            {"CLUSTERS0", OFFSETOF_NESTED(config_t, ivf_config, index_params, clusters0)}
                        }); 
            }) { }
    };
    class index : public X_IVF_PQ::index {
    public:
        index(pq::index_params const& pq_params, X_IVF_PQ::index_params const& secondary_config, std::string const& filter_config_path, uint64_t topk, 
            index_params ivf_config) : X_IVF_PQ::index(pq_params, secondary_config, filter_config_path, topk), ivf_config(ivf_config) {}
    protected:
        void build_first_level_index(
            build_input_t const& in
        ) override;
        void select_first_level_sets(
            query_input_t const& query_in, 
            raft::device_matrix_view<int, uint64_t> bitmap_matrix
        ) const override;
        uint64_t estimate_filter_ratio(
            query_input_t const& query_in    
        ) const override;
    private: 
        index_data_t ivf_params;
        index_params ivf_config;
    };
}