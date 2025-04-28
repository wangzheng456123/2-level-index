#include <search-engine/search-engine-auto-batch.cuh>
#include <iostream>

INITIALIZE_EASYLOGGINGPP
INIT_PARAFILTER_MMR_STATIC_MEMBERS

int main(int argc, char* argv[]) 
{
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <algorithm> <config_path> [filter_config_path]\n";
        return 1;
    }

    std::string algorithm = argv[1];
    std::string config_path = argv[2];
    std::string filter_config_path;
    if (argc == 4) {
        filter_config_path = argv[3];
    } else {
        filter_config_path = ""; 
    }

    auto_batch_search search_engine(algorithm, config_path, filter_config_path);
    search_engine.run();

    std::cout << "search finish";
    return 0;

}