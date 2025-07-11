#include <execinfo.h>
#include <limits>
#include <set>
#include <unordered_set>

inline void print_stack_trace() {
    void *buffer[10]; 
    int size;

    size = backtrace(buffer, 10);  

    char **symbols = backtrace_symbols(buffer, size);
    for (int i = 0; i < size; i++) {
        printf("%s\n", symbols[i]);
    }

    free(symbols);  
}

template<typename ElementType, typename IndexType>
bool sample_verification(ElementType *ground_truth, 
                         raft::device_matrix_view<ElementType, IndexType> res_mat,  
                         uint32_t sample_count = 100) {
    IndexType n_row = res_mat.extent(0);
    IndexType n_dim = res_mat.extent(1);

    ElementType* host_res = copy_matrix_to_host(res_mat);

    for (uint32_t i = 0; i < sample_count; i++) {
        uint32_t x = rand() % n_row;
        uint32_t y = rand() % n_dim;

        uint32_t id = x * n_dim + y;

        if (std::abs(host_res[id] - ground_truth[id]) > 1e-5) {
            LOG(INFO) << "program failed to pass semantic check for x = " << 
                x << "y = " << y << "and sample: " << i;
            print_stack_trace();
            exit(0);
        }
    }

    delete [] host_res;
}

inline uint64_t read_neighbors_file(const std::string& file_path, std::vector<uint64_t>& neighbors) 
{
    uint64_t valid_cnt = 0;
    std::ifstream neighbors_in(file_path, std::ios::binary);
    if (!neighbors_in) {
        std::cerr << "Error: Unable to open ground truth neighbors file: " << file_path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int neighbor;
    std::unordered_set<int> seen_neighbors;  // Set to store already seen neighbors

    while (neighbors_in.read(reinterpret_cast<char*>(&neighbor), sizeof(int))) {
        // Only store valid neighbors (>= 0) and ensure it's not already in the set
        if (neighbor >= 0 && seen_neighbors.find(neighbor) == seen_neighbors.end()) {
            neighbors.push_back(neighbor);
            valid_cnt++;
            seen_neighbors.insert(neighbor);  // Mark the neighbor as seen
        }
        else {
            // fixme: use other magic for very large dataset
            neighbors.push_back(12345678910ll);  // Use magic value for duplicate or invalid neighbors
        }
    }

    neighbors_in.close();
    return valid_cnt;
}

inline float compute_recall(const std::string& res_directory_path, const std::string& ground_truth_path, int topk, int n_queries) {
    // Step 1: Load results from the res_directory
    std::vector<uint64_t> res_neighbors;  // To store the final res_neighbors
    std::vector<float> res_distances;
    
    load_res_files_from_directory(res_directory_path, res_distances, res_neighbors);
    auto res_neighbors_ptr = res_neighbors.data();

    // Step 2: Load ground truth neighbors
    std::vector<uint64_t> neighbors;  // To store the ground truth neighbors
    uint64_t total_valid = read_neighbors_file(ground_truth_path, neighbors);
    uint64_t total_hits = 0;   // To store the total number of hits

    // Step 3: Ensure res_neighbors and neighbors are properly sized
    assert(res_neighbors.size() >= n_queries * topk);
    assert(neighbors.size() >= n_queries * topk);

    // Step 4: Compute accuracy
    for (int i = 0; i < n_queries; ++i) {
        // For each query, check topk elements
        std::unordered_set<uint64_t> valid_neighbors_set;
        
        // Collect valid neighbors for the current query
        for (int j = 0; j < topk; ++j) {
            int idx = i * topk + j;
            valid_neighbors_set.insert(neighbors[idx]);
        }

        // Now compare res_neighbors for the current query with valid ground truth neighbors
        int hits = 0;
        std::unordered_set<int> seen_neighbors;  

        for (int j = 0; j < topk; ++j) {
            int idx = i * topk + j;
            
            if (seen_neighbors.find(res_neighbors[idx]) != seen_neighbors.end()) {
                continue;
            }
            
            seen_neighbors.insert(res_neighbors[idx]);
            if (valid_neighbors_set.find(res_neighbors[idx]) != valid_neighbors_set.end()) {
                ++hits;
            }
        }

        if (hits < 50) {
            LOG(INFO) << i << "th query very low recall, with" << hits << "%";
        }
        // Update the totals
        total_hits += hits;
    }

    // Step 5: Compute and return accuracy
    if (total_valid == 0) return 0.0f;  // To prevent division by zero
    return static_cast<float>(total_hits) / total_valid;
}
