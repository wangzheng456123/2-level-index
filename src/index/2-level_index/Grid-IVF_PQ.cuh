#include <index/2-level_index/X-IVF_PQ.cuh>
namespace Grid_IVF_PQ {
    struct grid_meta {
        float len;
        uint64_t cnt;
        float start;
        float end;
    };

    struct index_data_t {
        raft::device_matrix_view<float, uint64_t> ranges;     
    };

    struct index_params {
        uint64_t max_grids_cnt;
    };

    struct config_t: X_IVF_PQ::config_t {
        index_params grids_config;

        config_t(std::string const& path):
            X_IVF_PQ::config_t(path, [](std::map<std::string, int> &m) {
               m.insert({ 
                            {"MAX_GRIDS_CNT", OFFSETOF_NESTED(config_t, grids_config, index_params, max_grids_cnt)}
                        }); 
            }) { }
    };

    class index : public X_IVF_PQ::index {
    public:
        index(pq::index_params const& pq_params, X_IVF_PQ::index_params const& secondary_config, std::string const& filter_config_path, uint64_t topk,
              index_params grids_config) : X_IVF_PQ::index(pq_params, secondary_config, filter_config_path, topk), grids_config(grids_config) {}
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
        void generate_grid_meta(build_input_t const& in, std::vector<grid_meta> &grid_info_host);
        index_params grids_config;
        index_data_t grid_params;
        // for many cases, filter ratio after griding can be estimated directly when building
        uint64_t estimated_filter_ratio = 0;
    };
}