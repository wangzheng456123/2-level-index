#pragma once
#include <index/2-level_index/IVF-IVF_PQ.cuh>
#include <index/2-level_index/Grid-IVF_PQ.cuh>
#include <index/parafiltering.cuh>
#include <index/prefiltering.cuh>
#include <index/product-quantization.cuh>
#include <index/sorting-index.cuh>

class config_factory {
public:
    static std::unique_ptr<common_config> create(
        std::string const& algorithm, 
        std::string const& search_config_path
    )
    {
        std::unique_ptr<common_config> config_ptr;
        if (algorithm == "pq" || algorithm == "prefilter" || algorithm == "sorting") 
            config_ptr = std::make_unique<pq::config_t>(search_config_path, [](auto&&...) {});
        else if (algorithm == "parafilter")
            config_ptr =  std::make_unique<parafiltering::config_t>(search_config_path);
        else if (algorithm == "ivf-ivf_pq")
            config_ptr = std::make_unique<IVF_IVF_PQ::config_t>(search_config_path);
        else if (algorithm == "grid-ivf_pq")
            config_ptr = std::make_unique<Grid_IVF_PQ::config_t>(search_config_path);
        else throw std::invalid_argument("Unknown index type: " + algorithm);
        config_ptr->read_config();
        return config_ptr;
    }
};

class index_factory {
public:
    static std::unique_ptr<index_base_t> create(
        std::string const& algorithm, 
        common_config* config_ptr,
        std::string const& filter_config_path
    ) 
    {
        uint64_t topk = config_ptr->topk;
        if (algorithm == "pq") {
            pq::config_t* pq_config_ptr = static_cast<pq::config_t*>(config_ptr);
            return std::make_unique<pq::index>(pq_config_ptr->pq_config, topk);
        }
        else if (algorithm == "prefilter") {
            pq::config_t* pq_config_ptr = static_cast<pq::config_t*>(config_ptr);
            return std::make_unique<prefiltering::index>(pq_config_ptr->pq_config, filter_config_path, topk);
        }
        else if (algorithm == "parafilter") {
            parafiltering::config_t* parafilter_config_ptr = static_cast<parafiltering::config_t*>(config_ptr);
            return std::make_unique<parafiltering::index>(parafilter_config_ptr->pq_config, 
                parafilter_config_ptr->parafilter_config, filter_config_path, topk);
        }
        else if (algorithm == "sorting") {
            pq::config_t* pq_config_ptr = static_cast<pq::config_t*>(config_ptr);
            return std::make_unique<sorting::index>(pq_config_ptr->pq_config, filter_config_path, topk);
        }
        else if (algorithm == "ivf-ivf_pq") {
            IVF_IVF_PQ::config_t* ivf_ivfpq_config_ptr = static_cast<IVF_IVF_PQ::config_t*>(config_ptr);
            return std::make_unique<IVF_IVF_PQ::index>(ivf_ivfpq_config_ptr->pq_config, 
                ivf_ivfpq_config_ptr->secondary_config, filter_config_path, topk, ivf_ivfpq_config_ptr->ivf_config);;
        }
        else if (algorithm == "grid-ivf_pq") {
            Grid_IVF_PQ::config_t* grid_ivfpq_config_ptr = static_cast<Grid_IVF_PQ::config_t*>(config_ptr);
            return std::make_unique<Grid_IVF_PQ::index>(grid_ivfpq_config_ptr->pq_config, 
                grid_ivfpq_config_ptr->secondary_config, filter_config_path, topk, grid_ivfpq_config_ptr->grids_config);
        }
        else throw std::invalid_argument("Unknown index type: " + algorithm);
    }
};