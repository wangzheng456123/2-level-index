#pragma once
#include <core/mat_operators.cuh>
#include <core/interfaces.cuh>
#include <core/mmr.cuh>
#include <utils/config_utils.h>
#include <raft/cluster/kmeans.cuh>
#include <stdint.h>
#include <string>
#include <functional>
#include <string>
#include <map>

namespace pq {
    struct index_data_t {
        raft::device_matrix_view<uint8_t, uint64_t> codebook;
        raft::device_matrix_view<float, uint64_t> centers;
    };

    struct index_params {
        uint64_t pq_dim;
        uint64_t n_clusters;
        uint64_t exps0;
        uint64_t pq_len;
    };

    struct config_t: common_config {
        // parameters for build pq index
        index_params pq_config;

        config_t(std::string const& path, std::function<void(std::map<std::string, int>&)> extender):
            common_config(path, [](std::map<std::string, int> &m) {
               m.insert({ 
                            {"PQ_DIM", OFFSETOF_NESTED(config_t, pq_config, index_params, pq_dim)}, 
                            {"N_CLUSTERS", OFFSETOF_NESTED(config_t, pq_config, index_params, n_clusters)},
                            {"EXPS0", OFFSETOF_NESTED(config_t, pq_config, index_params, exps0)}
                        }); 
            }) { extender(str_to_offset_map); }
    };

    class index : public index_base_t {
    public:
        index(index_params const& pq_config, uint64_t topk) : pq_config(pq_config), index_base_t(topk)  {}
        void build(const build_input_t& in) override;
        void query(const query_input_t& in, query_output_t& out) const override;
        // moving it to public to walk around nvcc constrain.
        void build_pq(const build_input_t& in); 
    protected:
        void query_pq(const query_input_t& in, valid_indices_t const& valid_indices, query_output_t& out) const;
        void calc_pq_dis(const query_input_t& in, raft::device_matrix_view<float, uint64_t> pq_dis, valid_indices_t const& valid_indices) const;
        index_data_t pq_index;
        // index configrations
        index_params pq_config;
    private:
        void calc_batched_L2_distance(
            raft::device_resources const& dev_resources,
            raft::device_matrix_view<float, uint64_t> const& queries,       
            raft::device_matrix_view<uint64_t, uint64_t> const& indices,
            // index_t pq_index,
            raft::device_matrix_view<float, uint64_t> dis,
            uint64_t query_batch_size = 1, 
            uint64_t data_batch_size = 1, 
            uint64_t n_clusters = 256,
            int64_t n_indices = -1
        ) const;
    };
}