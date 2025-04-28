#pragma once
#include <index/prefiltering.cuh>
#include <raft/distance/distance.cuh>

namespace X_IVF_PQ {
    struct first_level_index_t {
        std::vector<raft::device_vector_view<uint64_t, uint64_t>> clusters_list;
    };

    struct second_level_index_t {
        raft::device_vector_view<uint64_t*, uint64_t> secondary_clusters_ptr;
        raft::device_vector_view<uint64_t, uint64_t> secondary_clusters_list_len;
        raft::device_matrix_view<float, uint64_t> secondary_centers_list;
        raft::device_vector_view<uint64_t, uint64_t> inv_secondary_id_map;
        uint64_t min_cluster_len;
        uint64_t max_cluster_len;
    };

    struct index_data_t {
        first_level_index_t first_level_index;
        second_level_index_t second_level_index;
    };

    struct index_params {
        uint64_t clusters1;
        uint64_t n_list;
    };

    struct config_t: pq::config_t {
        index_params secondary_config;

        config_t(std::string const& path, std::function<void(std::map<std::string, int>&)> extender):
            pq::config_t(path, [](std::map<std::string, int> &m) {
               m.insert({ 
                            {"CLUSTERS1", OFFSETOF_NESTED(config_t, secondary_config, index_params, clusters1)}, 
                            {"N_LIST", OFFSETOF_NESTED(config_t, secondary_config, index_params, n_list)}
                        }); 
            }) { extender(str_to_offset_map); }
    };

    class index : public prefiltering::index {
    public:
        index(pq::index_params const& pq_params, index_params const& secondary_config, std::string const& filter_config_path, uint64_t topk) :
            prefiltering::index(pq_params, filter_config_path, topk), secondary_config(secondary_config) {}
        void build(build_input_t const&) override;
        void query(query_input_t const&, query_output_t &) const override;
    protected:
        virtual void build_first_level_index(
            build_input_t const& in
        ) = 0;
        void build_second_level_index(
            build_input_t const& in
        );  
        virtual void select_first_level_sets(
            query_input_t const& query_in, 
            raft::device_matrix_view<int, uint64_t> bitmap_matrix
        ) const = 0;
        virtual uint64_t estimate_filter_ratio(
            query_input_t const& query_in    
        ) const = 0;
        void select_indices(
            query_input_t const& query_in, 
            raft::device_matrix_view<int, uint64_t> const& bitmap_matrix,
            raft::device_matrix_view<uint64_t, uint64_t> &indices 
        ) const; 
        void group_by_cluster_id(
            raft::device_vector_view<uint64_t, uint64_t> const& cluster_ids, 
            std::vector<raft::device_vector_view<uint64_t, uint64_t>>& grouped_data
        ); 
        index_data_t secondary_index_data;
        index_params secondary_config;
    private:
    };
}