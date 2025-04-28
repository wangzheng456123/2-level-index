#include <index/prefiltering.cuh>

namespace prefiltering {
    void index::build_filters_and_labels(build_input_t const& in)
    {   
        filters_and_labels.data_labels = in.data_labels;
    }

    void index::build(build_input_t const&in) {
        build_pq(in);
        build_filters_and_labels(in);
    }

    void index::prefilter(query_input_t const& in, valid_indices_t const& valid_indices_in, 
                          valid_indices_t &valid_indices_out) const
    {
        uint64_t n_queries = in.queries.extent(0);
        uint64_t n_dim = in.queries.extent(1);
        uint64_t l = f_config.l;

        auto ranges = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, 2 * l);
        recovery_filters(in.query_labels, ranges, f_config);

        bool filterd = valid_indices_in.valid_cnt >= 0 ? true : false;

        valid_indices_out.valid_cnt = filter_valid_data(
            in.dev_resources, 
            filters_and_labels.data_labels,
            ranges, 
            valid_indices_out.indices,
            valid_indices_in.indices, 
            filterd
        );
    }

    void index::query_prefilter(query_input_t const &in, 
                                valid_indices_t const& valid_indices_in, 
                                query_output_t &out) const
    {
        valid_indices_t valid_indices{};
        uint64_t n_data = filters_and_labels.data_labels.extent(0);
        uint64_t n_queries = in.queries.extent(0);

        if (valid_indices_in.valid_cnt >= 0) 
            valid_indices.indices = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, valid_indices_in.valid_cnt);
        else valid_indices.indices = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, n_data);
        
        prefilter(in, valid_indices_in, valid_indices); 

        query_pq(in, valid_indices, out);
    }

    void index::query(query_input_t const &in, 
                      query_output_t &out) const
    {   
        uint64_t n_queries = in.queries.extent(0);

        query_output_t tmp_out;
        tmp_out.neighbors = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, topk * pq_config.exps0);
        tmp_out.distances = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, topk * pq_config.exps0);

        valid_indices_t tmp_indices;
        tmp_indices.valid_cnt = -1;

        query_prefilter(in, tmp_indices, tmp_out);

        refine(in, tmp_out, out);
    }
}