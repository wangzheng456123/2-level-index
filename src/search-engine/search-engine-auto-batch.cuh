#include <search-engine/search-engine-interfaces.cuh>
#include <core/merge_res.cuh>
#include <index/index-factory.cuh>
#include <utils/io_utils.cuh>
#include <fstream>
#include <utils/perf_utils.cuh>
#include <verification/verification.cuh>

class auto_batch_search : public search_engine_base {
public:
    auto_batch_search(
        std::string const& algorithm, 
        std::string const& config_path,
        std::string const& filter_config_path
    );
    ~auto_batch_search() = default;
    void run() override;
protected:
    void build_dataset(std::string, uint64_t, uint64_t, uint64_t, uint64_t, build_input_t&, query_input_t&);
private:
    void calc_mem_coeff(double *coeff);
    void split_task(
        double* coeff,
        uint64_t &query_batch_size, 
        uint64_t &data_batch_size, 
        uint64_t aditional = 0, 
        uint64_t lowest_query_batch_size = 125) ;
     // task meta data
    uint64_t n_data;
    uint64_t n_queries;
    uint64_t n_dim;
    // number of filters
    uint64_t l;
    // dimension per data label
    uint64_t label_dim;
    // this is either 1 or 2
    int filter_dim;
    std::unique_ptr<index_base_t> index;
    std::unique_ptr<common_config> config;
    std::map<std::string, std::string> types;
    std::vector<std::string> keys;
    std::map<std::string, std::pair<int32_t, int32_t>> size_map; 
    filter_config f_config;
    const int SWAP_BUFF_COUNT = 2;
    // perf data
    float query_time; 
    float build_time;
};

