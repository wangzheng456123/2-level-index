#pragma once
#include <raft/core/device_mdspan.hpp>
#include <core/refine.cuh>
#include <stdint.h>

struct build_input_t { 
    raft::device_resources dev_resources;
    raft::device_matrix_view<float, uint64_t> dataset;
    raft::device_matrix_view<float, uint64_t> data_labels;
};
struct query_input_t { 
    raft::device_resources dev_resources;
    // only used when algorithm needs refine
    raft::device_matrix_view<float, uint64_t> dataset;
    raft::device_matrix_view<float, uint64_t> queries;
    raft::device_matrix_view<float, uint64_t> query_labels;
};
struct valid_indices_t {
    int64_t valid_cnt = -1;
    raft::device_matrix_view<uint64_t, uint64_t> indices;
};
struct query_output_t { 
    raft::device_matrix_view<float, uint64_t> distances;
    raft::device_matrix_view<uint64_t, uint64_t> neighbors;
};

class index_base_t {
public:
    index_base_t(uint64_t topk) : topk(topk) {} 
    virtual void build(const build_input_t&) = 0;
    virtual void query(const query_input_t&, query_output_t&) const = 0;
    virtual ~index_base_t() {}
    uint64_t topk;
protected:
    /*A simple wrapper for refine in all index*/
    void refine(const query_input_t& in, const query_output_t& tmp_out, query_output_t& out) const {
        ::refine(
            in.dev_resources,
            in.dataset, 
            in.queries,
            tmp_out.neighbors, 
            out.neighbors, 
            out.distances
        );
    }
};