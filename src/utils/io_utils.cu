
#include <utils/io_utils.cuh>

void get_data_set_list(std::vector<std::string> &keys, const char file[]) 
{
  std::ifstream fkeys(file, std::ios::binary);
  if (!fkeys.is_open()) {
    throw std::runtime_error("Failed to open file");
  }

  std::string key;
  while (std::getline(fkeys, key)) {
    keys.push_back(key);
    std::cout << "add input dataset: " << key << "\n";
  }
}

void get_data_type_list(std::map<std::string, std::string> &data_type, const char file[], 
                               std::vector<std::string> const& keys) 
{
  std::ifstream ftypes(file, std::ios::binary);
  std::string type;
  
  // types and key has an inner correspondence when build
  for (auto key : keys) {
    assert(std::getline(ftypes, type));
    data_type[key] = type;
  }
}

static size_t get_datatype_size(std::string const& type) 
{
  if (type == "int32") return sizeof(int32_t);
  else if (type == "float32") return sizeof(float);
  else if (type == "int64") return sizeof(int64_t);
  else if (type == "float64") return sizeof(double);
  else if (type == "int8") return sizeof(int8_t);
  else if (type == "uint8") return sizeof(uint8_t);
  else if (type == "uint16") return sizeof(uint16_t);
  else if (type == "uint32") return sizeof(uint32_t);
  else if (type == "uint64") return sizeof(uint64_t);
  else if (type == "char") return sizeof(char);
  else if (type == "bool") return sizeof(bool);
  else if (type == "int") return sizeof(int);
  else if (type == "long") return sizeof(long);
  else if (type == "long long") return sizeof(long long);
  else if (type == "unsigned int") return sizeof(unsigned int);
  else if (type == "unsigned long") return sizeof(unsigned long);
  else if (type == "unsigned long long") return sizeof(unsigned long long);
  else if (type == "short") return sizeof(short);
  else if (type == "unsigned short") return sizeof(unsigned short);
  else {
    LOG(INFO) << "data type not suported: " << type;
    exit(0);
  }
}

void* read_binary_file(const std::string& file_path, uint64_t offset, uint64_t size) {
    std::ifstream file(file_path.c_str(), std::ios::binary);
    if (!file) {
        LOG(ERROR) << "Failed to open file: " << file_path;
        return nullptr;
    }

    file.seekg(offset);
    void* buffer = malloc(size);
    if (!buffer) {
        LOG(ERROR) << "Failed to allocate memory for file: " << file_path;
        return nullptr;
    }

    file.read(reinterpret_cast<char*>(buffer), size);
    uint64_t read_size = file.gcount();

    // Zero out any remaining memory if not fully read
    if (read_size < size) {
        std::memset(static_cast<char*>(buffer) + read_size, 0, size - read_size);
    }

    return buffer;
}

std::future<bool> write_binary_file_async(const std::string& file_path, uint64_t offset, const void* data, uint64_t size, bool append) {
    // Launch an asynchronous task to handle file writing
    return std::async(std::launch::async, [file_path, offset, data, size, append]() -> bool {

        // Determine the open mode based on the append parameter
        std::ios::openmode mode = std::ios::binary;
        if (append) {
            mode |= std::ios::app;
        }

        // If not appending and the file exists, delete the existing file
        if (!append) {
            std::ifstream existing_file(file_path);
            if (existing_file) {
                existing_file.close(); // Close the file before deleting
                if (std::remove(file_path.c_str()) != 0) {
                    std::cerr << "Failed to delete existing file: " << file_path << std::endl;
                    return false;
                }
            }
        }

        // Open the file with the appropriate mode
        std::ofstream file(file_path.c_str(), mode);

        // Try opening the file
        if (!file) {
            std::cerr << "Failed to open or create file: " << file_path << std::endl;
            return false;
        }

        // If in append mode, ignore the offset and write at the end of the file
        if (!append) {
            // Seek to the specified offset
            file.seekp(offset);
            if (!file) {
                std::cerr << "Failed to seek to offset in file: " << file_path << std::endl;
                return false;
            }
        }

        // Write data to the file
        file.write(reinterpret_cast<const char*>(data), size);
        if (!file) {
            std::cerr << "Failed to write data to file: " << file_path << std::endl;
            return false;
        }

        return true;
    });
}

void load_res_files_from_directory(const std::string& directory_path, 
                                std::vector<float>& distances,
                                std::vector<uint64_t>& neighbors) {
    // Temporary storage for file paths
    std::vector<std::string> distance_files;
    std::vector<std::string> neighbor_files;

    // Traverse the directory and collect file names
    for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
        if (entry.is_regular_file()) {
            const std::string file_name = entry.path().filename().string();
            if (file_name.find("distances") == 0) {
                distance_files.push_back(entry.path().string());
            } else if (file_name.find("neighbors") == 0) {
                neighbor_files.push_back(entry.path().string());
            }
        }
    }

    // Sort file names in lexicographical order
    std::sort(distance_files.begin(), distance_files.end());
    std::sort(neighbor_files.begin(), neighbor_files.end());

    // Read and append data from distance files
    for (const auto& file : distance_files) {
        std::ifstream in(file, std::ios::binary | std::ios::ate);
        if (!in) {
            std::cerr << "Failed to open file: " << file << std::endl;
            continue;
        }

        size_t file_size = in.tellg();
        size_t num_elements = file_size / sizeof(float);

        // Allocate memory for the data and read it
        float* buffer = static_cast<float*>(read_binary_file(file, 0, file_size));
        if (buffer) {
            size_t old_size = distances.size();
            distances.resize(old_size + num_elements);
            std::memcpy(distances.data() + old_size, buffer, file_size);
            free(buffer); // Free the allocated memory
        }
    }

    // Read and append data from neighbor files
    for (const auto& file : neighbor_files) {
        std::ifstream in(file, std::ios::binary | std::ios::ate);
        if (!in) {
            std::cerr << "Failed to open file: " << file << std::endl;
            continue;
        }

        size_t file_size = in.tellg();
        size_t num_elements = file_size / sizeof(uint64_t);

        // Allocate memory for the data and read it
        uint64_t* buffer = static_cast<uint64_t*>(read_binary_file(file, 0, file_size));
        if (buffer) {
            size_t old_size = neighbors.size();
            neighbors.resize(old_size + num_elements);
            std::memcpy(neighbors.data() + old_size, buffer, file_size);
            free(buffer); // Free the allocated memory
        }
    }
}

void build_dataset(
    std::vector<std::string> const& keys,
    std::map<std::string, std::string> const& data_type,
    std::map<std::string, void*>* data_map,
    std::map<std::string, std::pair<int32_t, int32_t>>& size_map,
    const std::string &dir,
    uint64_t data_offset,
    uint64_t data_batch_size,
    uint64_t query_offset,
    uint64_t query_batch_size, 
    int filter_dim) 
{
    for (const auto& key : keys) {

        // Load size map if not already provided
        if (data_batch_size == 0) {
            std::string size_path = dir + key + "_size";
            void* size_data = read_binary_file(size_path, 0, sizeof(int32_t) * 2);
            if (!size_data) continue;

            int32_t* sizes = reinterpret_cast<int32_t*>(size_data);
            size_map[key] = {sizes[0], sizes[1]};
            free(size_data);
        }

        if (data_map) {
            int64_t n_dim = size_map[key].second;
            int64_t n_row = 0;
            uint64_t offset = 0;

            // Thread-local state for offset tracking
            thread_local uint64_t current_data_offset = 0;
            thread_local uint64_t current_query_offset = 0;
            thread_local uint64_t current_data_size = 0;
            thread_local uint64_t current_query_size = 0;
            thread_local uint64_t current_data_label_size = 0;
            thread_local uint64_t current_data_label_offset = 0;
            thread_local uint64_t current_query_label_size = 0;
            thread_local uint64_t current_query_label_offset = 0;

            auto end_in_advance = [&offset](uint64_t data_read_size, uint64_t data_read_offset,
                                            uint64_t label_read_size, uint64_t label_read_offset, bool is_label) -> bool {
                if (!is_label) {
                    return data_read_size && data_read_offset == offset;
                } else {
                    return label_read_size && label_read_offset == offset;
                }
            };

            auto modify_read_state = [&offset, &n_row](uint64_t& data_read_size, uint64_t& data_read_offset,
                                                      uint64_t& label_read_size, uint64_t& label_read_offset, bool is_label) {
                if (is_label) {
                    label_read_size = n_row;
                    label_read_offset = offset;
                } else {
                    data_read_size = n_row;
                    data_read_offset = offset;
                }
            };

            bool is_label = (key.find("label") != std::string::npos);

            if (key.find("train") != std::string::npos) {
                n_row = data_batch_size;
                offset = data_offset;

                if (end_in_advance(current_data_size, current_data_offset,
                                   current_data_label_size, current_data_label_offset, is_label)) {
                    continue;
                }
                modify_read_state(current_data_size, current_data_offset,
                                  current_data_label_size, current_data_label_offset, is_label);

            } else if (key.find("test") != std::string::npos) {
                n_row = query_batch_size;
                if (is_label) n_dim *= filter_dim; 

                offset = query_offset;

                if (end_in_advance(current_query_size, current_query_offset,
                                   current_query_label_size, current_query_label_offset, is_label)) {
                    continue;
                }
                modify_read_state(current_query_size, current_query_offset,
                                  current_query_label_size, current_query_label_offset, is_label);
            } else {
                continue;
            }

            LOG(TRACE) << "build data set: " << key << " with: "
                      << n_row << " rows, " << n_dim << " dimensions from offset: "
                      << offset;

            size_t element_size = get_datatype_size(data_type.at(key));
            uint64_t read_size = n_row * n_dim * element_size;
            std::string data_path = dir + key;
            void* data = read_binary_file(data_path, offset * n_dim * element_size, read_size);
            if (!data) continue;

            void* device_data;
            if (!(data_map->count(key))) {
                device_data = parafilter_mmr::mem_allocator(read_size);
                (*data_map)[key] = device_data;
            } else {
                device_data = (*data_map)[key];
            }

            cudaMemcpy(device_data, data, read_size, cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("cudaMemcpy failed");
            free(data);
        }
    }
}
