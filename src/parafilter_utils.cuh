#pragma once
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/random/sample_without_replacement.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdint>
#include <fstream>
#include <future> 
#include <filesystem>

// #define ELPP_THREAD_SAFE 1
#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
/**
* Check for CUDA errors; print and exit if there was a problem.
*/
inline void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      LOG(ERROR) << "Line :" << line;
    }
    LOG(ERROR) << "Cuda error :" << msg << ":" << cudaGetErrorString(err);
    exit(EXIT_FAILURE);
  }
}

inline void export_fvecs_file(float* data, uint64_t n_data, uint64_t n_dim, const char file_name[])  
{
    std::ofstream out(file_name, std::ios::binary | std::ios::app);
    if (!out) {
        throw std::runtime_error("Failed to open output file.");
    }

    for (uint64_t i = 0; i < n_data; ++i) {
        int32_t dim = static_cast<int32_t>(n_dim);
        out.write(reinterpret_cast<const char*>(&dim), sizeof(int32_t));

        out.write(reinterpret_cast<const char*>(data + i * n_dim), sizeof(float) * n_dim);
    }

    out.close();
}

inline void get_current_device_mem_info(uint64_t &available, uint64_t &total) {
  int id;
  cudaGetDevice(&id);
  cudaMemGetInfo(&available, &total);
}

inline uint32_t get_cur_device_maxi_threads() {
  int id;
  cudaGetDevice(&id);
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, id);
  return devProp.multiProcessorCount * devProp.maxThreadsPerMultiProcessor;
}

// todo: put these helpers to a seperate file.
inline
std::string& toUpper(std::string& str) 
{
  std::transform(str.begin(), str.end(), str.begin(),
  [](char c) {
    return static_cast<char>(::toupper(c));
  });
    return str;
}
inline
bool contains(const char* str, char c) {
  for (; *str; ++str) {
    if (*str == c)
      return true;
  }
  return false;
}
inline
std::string& ltrim(std::string& str) {
  str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](char c) {
  return !std::isspace(c);
  } ));
  return str;
}
inline
std::string& rtrim(std::string& str) {
  str.erase(std::find_if(str.rbegin(), str.rend(), [](char c) {
    return !std::isspace(c);
  }).base(), str.end());
  return str;
}
inline
std::string& trim(std::string& str) {
  return ltrim(rtrim(str));
}
inline
bool isInteger(const std::string& str) {
  if (str.empty()) return false;

  size_t i = 0;
  if (str[i] == '+' || str[i] == '-') {
      i++;
  }

  for (; i < str.size(); i++) {
      if (!isdigit(str[i])) {
          return false;
      }
  }
  return i > 0;
}

inline
std::string remove_spaces(const std::string& str) {
  std::string result;
  for (char ch : str) {
    if (!isspace(ch)) {
      result += ch;
    }
  }
  return result;
}

template <typename T>
std::vector<T> parse_array(const std::string& str) {
  std::vector<T> result;
  std::string cleaned_str = str;

  if (!cleaned_str.empty() && cleaned_str.front() == '[') {
    cleaned_str.erase(0, 1); 
  }
  if (!cleaned_str.empty() && cleaned_str.back() == ']') {
    cleaned_str.pop_back(); 
  }

  std::stringstream ss(cleaned_str);
  std::string token;

  while (std::getline(ss, token, ',')) {
    token.erase(0, token.find_first_not_of(" \t"));
    token.erase(token.find_last_not_of(" \t") + 1);

    if (!token.empty()) {
      result.push_back(static_cast<T>(std::stod(token))); 
    }
  }

  return result;
}

struct parafilter_config {
  uint64_t data_width;
  uint64_t index_width;
  // parameters for build pq index
  uint64_t pq_dim;
  uint64_t n_clusters;
  // parameters for multiple rounds filtering
  uint64_t exp1;
  uint64_t exp2;
  uint64_t topk;
  uint64_t break_down;
  uint64_t enable_multi_gpu;
  uint64_t mem_bound;
  uint64_t is_calc_mem_predictor_coeff;
  uint64_t lowest_query_batch;
  uint64_t filter_dim;
  float merge_rate;

  std::string dataset;
  std::string path;

  parafilter_config(std::string const& path_to_config) {
    std::ifstream fileStream_(path_to_config.c_str(), std::ifstream::in);
    std::string line = std::string();

    while (fileStream_.good()) {
      std::getline(fileStream_, line);
      std::size_t assignment = line.find('=');
      std::string currConfigStr = line.substr(0, assignment);
      currConfigStr = toUpper(currConfigStr);
      currConfigStr = trim(currConfigStr);
      // currConfig = ConfigurationTypeHelper::convertFromString(currConfigStr->c_str());
      std::string currValue = line.substr(assignment + 1);
      currValue = trim(currValue);
      std::size_t quotesStart = currValue.find("\"", 0);
      std::size_t quotesEnd = std::string::npos;
      if (quotesStart != std::string::npos) {
        quotesEnd = currValue.find("\"", quotesStart + 1);
        while (quotesEnd != std::string::npos && currValue.at(quotesEnd - 1) == '\\') {
          currValue = currValue.erase(quotesEnd - 1, 1);
          quotesEnd = currValue.find("\"", quotesEnd + 2);
        }
      }
      if (quotesStart != std::string::npos && quotesEnd != std::string::npos) {
        // Quote provided - check and strip if valid
        assert(quotesStart < quotesEnd);
        // assert(quotesStart + 1 != quotesEnd);
        if ((quotesStart != quotesEnd) && (quotesStart + 1 != quotesEnd)) {
        // Explicit check in case if assertion is disabled
          currValue = currValue.substr(quotesStart + 1, quotesEnd - 1);
        }
      }

      if (str_to_offset_map.find(currConfigStr) == str_to_offset_map.end()) {
        LOG(ERROR) << "ERROR: Key '" << currConfigStr << "' not found in str_to_offset_map!";
        continue;
      }
    
      if (quotesStart == std::string::npos) {
        if (isInteger(currValue)) {
            *(static_cast<uint64_t *>((static_cast<void*>(this) + str_to_offset_map[currConfigStr])))
                = (uint64_t)(std::atoll(currValue.c_str()));
        }
        else {
            *(static_cast<float *>((static_cast<void*>(this) + str_to_offset_map[currConfigStr])))
                = (float)(std::stof(currValue.c_str()));
        }
      }
      else {
        *(static_cast<std::string *>((static_cast<void*>(this) +  str_to_offset_map[currConfigStr])))
            = currValue;
      }
    }
  }

  private:
  /*initialize string to offset data in class static function*/
  static std::map<std::string, int> str_to_offset_map;
};

class filter_config {
public:
  int l;
  std::vector<int> filter_type;
  std::vector<float> shift_val;
  std::vector<int> div_value;
  std::vector<std::vector<int>> interval_map;

  filter_config(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Unable to open configuration file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
      line = remove_spaces(line);

      if (line.empty() || line[0] == '#') {
          continue;
      }

      if (line.find("l=") == 0) {
          l = std::stoi(line.substr(2));
          filter_type.resize(l);
          shift_val.resize(2 * l);
          interval_map.resize(l);
          div_value.resize(l);
      }
      else if (line.find("filter") == 0) {
        int filter_index = std::stoi(line.substr(6));

        while (std::getline(file, line)) {
          line = remove_spaces(line);
          if (line.empty()) break;

          if (line.find("type=") == 0) {
            int filter = std::stoi(line.substr(5));
            filter_type[filter_index] = filter;
            if (filter == 3) break;
          }
          else if (line.find("shift_val=") == 0) {
            std::vector<float> interval = parse_array<float>(line.substr(10));
            shift_val[2 * filter_index] = interval[0];
            shift_val[2 * filter_index + 1] = interval[1];
            break;
          }
          else if (line.find("interval_map=") == 0) {
            interval_map[filter_index] = parse_array<int>(line.substr(13));
            break;
          }
          else if (line.find("div_value=") == 0) {
            std::vector<int> div = parse_array<int>(line.substr(10));
            div_value[filter_index] = div[0];
            break;
          }
        }
      }
    }
  }

  void print_config() const {
    std::cout << "l = " << l << std::endl;
    for (int i = 0; i < l; ++i) {
      std::cout << "Filter " << i << ":" << std::endl;
      std::cout << "  Type: " << filter_type[i] << std::endl;
      std::cout << "  Shift Values: ";
      std::cout << shift_val[2 * i] << " " << shift_val[2 * i + 1] << " ";

      std::cout << std::endl;

      std::cout << "  Interval Map: ";
      for (int val : interval_map[i]) {
          std::cout << val << " ";
      }
      std::cout << std::endl;
    }
  }

private:

};

void process_filter_config(
    const filter_config& config,
    std::vector<float>& shift_len,
    std::vector<std::vector<float>>& maps_len) 
{
  if (config.shift_val.size() != 2 * config.l) {
    throw std::invalid_argument("shift_val size must be 2 * l");
  }

  shift_len.clear();
  for (size_t i = 0; i < config.l; ++i) {
    float left = config.shift_val[i * 2];
    float right = config.shift_val[i * 2 + 1];
    shift_len.push_back(std::abs(right + left));
  }

  maps_len.clear();
  for (const auto& intervals : config.interval_map) {
    std::vector<float> invervals_len;
    size_t n_points = intervals.size() / 2;

    for (size_t j = 0; j < n_points; ++j) {
      float l = intervals[j * 2];
      float r = intervals[j * 2 + 1];
      invervals_len.push_back(r - l);
    }
    maps_len.push_back(invervals_len);
  }
}

class Timer {
  public:
  Timer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }
  void start_timer()
  {
    cudaEventRecord(start);
  }
  void stop_timer()
  {
    cudaEventRecord(stop);
  }

  float get_time()
  {
    cudaError_t status;
    float millisec = 0;
    cudaEventSynchronize(stop);
    status = cudaEventElapsedTime(&millisec, start, stop);
    return millisec;
  }

  private:
  cudaEvent_t start;
  cudaEvent_t stop;
};

template<typename ElementType, typename IndexType>
int gauss(ElementType a[][5], IndexType n = 4)
{
    int r,c;  
    double eps = 1e-30;
    for(r=0,c=0;c<n;c++)   
    {
        int t=r;
        for(int i=r;i<n;i++)                       
            if(fabs(a[i][c])>fabs(a[t][c]))
                t=i;
        if(fabs(a[t][c])<eps) continue;    
        for(int i=c;i<=n;i++) std::swap(a[t][i],a[r][i]);   
        for(int i=n;i>=c;i--) a[r][i] /=a[r][c];   
        for(int i=r+1;i<n;i++)             
            if(fabs(a[i][c])>eps)
            {
                for(int j=n;j>=c;j--)
                    a[i][j]-=a[i][c]*a[r][j];
            }
        r++;
    }
    if(r<n)     
    {
        for(int i=r;i<n;i++)
            if(fabs(a[i][n])>eps)     
                return 2;
        return 1;        
    }
    for(int i=n-1;i>=0;i--)               
        for(int j=i+1;j<n;j++)
            a[i][n]-=a[j][n]*a[i][j];
    return 0;
}

int findMaxFactor(uint64_t a, uint64_t b) {
    if (b <= a) {
        return b; 
    }

    int maxFactor = 1;

    for (int i = 1; i <= sqrt(b); ++i) {
        if (b % i == 0) {
            if (i < a) {
                maxFactor = max(maxFactor, i);
            }
            int pairFactor = b / i;
            if (pairFactor < a) {
                maxFactor = max(maxFactor, pairFactor);
            }
        }
    }

    return maxFactor;
}


//fixme: fake run use macro is not simple to control the, 
//call in any levels, try to find better way 
// todo: Add cudaDeviceSyncronize for kernel break-down anylyze.
#define parafilterPerfLogWraper(func, time) \
  if (break_down) { \
    global_timer.start_timer(); \
    {func ;} \
    global_timer.stop_timer(); \
    float elapsed = global_timer.get_time(); \
    time += elapsed; \
    LOG(INFO) << __func__ << " " << __LINE__ << " elapsed with time: " << elapsed / 1000.f << "s"; \
  } \
  else { \
    {func ;} \
  } 


/*A simple warper for raft make device matrix call, used to trace app usage for raft resrouce*/
class parafilter_mmr {
  public:
  static void init_mmr() {
    int device_count;
    cudaGetDeviceCount(&device_count);

    total.resize(device_count);
    available.resize(device_count);
    cur_mems.resize(device_count);
    cur_offset.resize(device_count);
  }

  static uint64_t get_current_workspace_free() 
  {
    int id;
    cudaGetDevice(&id);
    return available[id];
  }   

  static uint64_t get_current_workspace_total() 
  {
    int id;
    cudaGetDevice(&id);
    return total[id];
  }
  static uint64_t get_current_workspace_used() {
    int id;
    cudaGetDevice(&id);
    return total[id] - available[id];
  }

  static void reset_current_workspace(uint64_t size) {
    int id;
    cudaGetDevice(&id);
    total[id] = size; 
    available[id] = size; 
  }

  static void set_fakeness() {
    fake_run = true;
  }

  static void clear_fakeness() {
    fake_run = false;
  }

  template<typename ElementType, typename IndexType>
  static auto make_device_matrix_view(IndexType n_row, 
                                      IndexType n_dim) 
  {
    ElementType *device_ptr;
    uint64_t size = sizeof(ElementType) * n_row * n_dim;
    if (!fake_run) {
      device_ptr = (ElementType*)mem_allocator(size);
      LOG(INFO) << "alloc memory with size: " << size;
      return raft::make_device_matrix_view<ElementType, IndexType>(device_ptr, n_row, n_dim);
    }
    else {
      // nothing will happen and a null mdspan will returned
      LOG(INFO) << "fake alloc memory with size: " << size;
      return raft::make_device_matrix_view<ElementType, IndexType>(nullptr, n_row, n_dim);
    }
  }
  
  template<typename ElementType, typename IndexType>
  static auto make_device_vector_view(IndexType n_elements) {
    ElementType *device_ptr;
    uint64_t size = sizeof(ElementType) * n_elements;
    if (!fake_run) {
      device_ptr = (ElementType*)mem_allocator(size);
      LOG(INFO) << "alloc memory with size: " << size;
      return raft::make_device_vector_view<ElementType, IndexType>(device_ptr, n_elements);
    }
    else {
      // nothing will happen and a null mdspan will returned
      LOG(INFO) << "fake alloc memory with size: " << size;
      return raft::make_device_vector_view<ElementType, IndexType>(nullptr, n_elements);
    }
  }

  static void workspace_add_mem(void *mem, uint64_t size) {
    int id;
    cudaGetDevice(&id);
    available[id] -= size;
    LOG(INFO) << "add device mem with" << mem;
    cur_mems[id][size].push_back(mem);
  }

  static void* mem_allocator(uint64_t size) {
    int id;
    cudaGetDevice(&id);
    LOG(INFO) << "allocate memory with " << size << " byte on device :" << id; 
    if (id < cur_mems.size() && cur_mems[id].count(size) && cur_offset[id][size] < cur_mems[id][size].size()) {
      int offset = cur_offset[id][size];
      cur_offset[id][size]++;
      LOG(INFO) << "parafilter mmr allocate block from pool";
      return cur_mems[id][size][offset];
    }
    else {
      void* mem;
      cudaMalloc((void**)&mem, size);
      LOG(INFO) << "parafilter mmr allocate block runtime";
      checkCUDAErrorWithLine("cudaMalloc failed");
      workspace_add_mem(mem, size);
      cur_offset[id][size]++;
      LOG(INFO) << cur_mems[id][size].size() << " blocks, " << cur_offset[id][size]
        << " block offset";
      return mem;
    }
  }

  static void free_cur_workspace_device_mems(bool free_mems = true) {
    int id;
    cudaGetDevice(&id);
    for (auto iter = cur_offset[id].begin(); iter != cur_offset[id].end(); ++iter) {
      iter->second = 0;
    }
    if (free_mems) {
      for (auto iter = cur_mems[id].begin(); iter != cur_mems[id].end(); ++iter) {
        auto key = iter->first;
        for (auto _iter = cur_mems[id][key].begin(); _iter != cur_mems[id][key].end(); ++_iter) { 
          LOG(INFO) << "free device mem with" << *_iter;
          cudaFree(*_iter);
          checkCUDAErrorWithLine("free work space memory failed");
        }
        cur_mems[id][key].clear();
      }
      cur_mems[id].clear();
    }
  }

  // todo: make these data thread safe
  static bool fake_run;
  static std::vector<uint64_t> total;
  static std::vector<uint64_t> available;
  // todo: implement mmr deconstructor to avoid GPU mem leak 
  static std::vector<std::map<uint64_t, std::vector<void *>>> cur_mems;
  static std::vector<std::map<uint64_t, int>> cur_offset;
   ~parafilter_mmr();
  private:
};

/*helpers for print raft device view*/
inline void get_data_set_list(std::vector<std::string> &keys, const char file[]) 
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

inline void get_data_type_list(std::map<std::string, std::string> &data_type, const char file[], 
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

inline size_t get_datatype_size(std::string const& type) 
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

inline void* read_binary_file(const std::string& file_path, uint64_t offset, uint64_t size) {
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

template <typename ElementType, typename IndexType>
void read_matrices_from_file(const std::string& file_path, IndexType n_queries, int topk,
                             IndexType batch_size, std::vector<std::vector<ElementType>>& matrices) 
{
    IndexType matrix_size = n_queries * topk * sizeof(ElementType);
    IndexType batch_data_size = matrix_size * batch_size;

    matrices.clear();
    matrices.resize(batch_size);

    void* data = nullptr;
    try {
        data = read_binary_file(file_path, 0, batch_data_size);
        ElementType* matrix_data = static_cast<ElementType*>(data);
        for (uint64_t i = 0; i < batch_size; ++i) {
            std::vector<ElementType> matrix(matrix_data + i * n_queries * topk,
                                            matrix_data + (i + 1) * n_queries * topk);
            matrices[i] = std::move(matrix);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error while reading matrices: " << e.what() << std::endl;
        if (data) {
            free(data);
        }
        throw;
    }

    if (data) {
        free(data);
    }
}

inline std::future<bool> write_binary_file_async(const std::string& file_path, uint64_t offset, const void* data, uint64_t size, bool append = true) {
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

template <typename ElementType, typename IndexType>
void merge_matrices_to_gpu(const std::vector<std::vector<ElementType>>& matrices, ElementType* d_merged_matrix, 
                           IndexType n_queries, int topk, IndexType batch_size) 
{
    IndexType merged_cols = batch_size * topk;

    for (IndexType b = 0; b < batch_size; ++b) {
        const auto& matrix = matrices[b];
        for (IndexType i = 0; i < n_queries; ++i) {
            cudaMemcpy(d_merged_matrix + i * merged_cols + b * topk, 
                       matrix.data() + i * topk, 
                       topk * sizeof(ElementType), 
                       cudaMemcpyHostToDevice);
        }
    }
}

inline void build_dataset(
    std::vector<std::string> const& keys,
    std::map<std::string, std::string> const& data_type,
    std::map<std::string, void*>* data_map,
    std::map<std::string, std::pair<int32_t, int32_t>>& size_map,
    const std::string &dir,
    uint64_t data_offset = 0,
    uint64_t data_batch_size = 0,
    uint64_t query_offset = 0,
    uint64_t query_batch_size = 0, 
    int filter_dim = 1) 
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

            std::string fvecs = "fvecs_data/" + key + ".fvecs";
            
            export_fvecs_file(static_cast<float*>(data), n_row, n_dim, fvecs.c_str());
            
            if (!data) continue;

            void* device_data;
            if (!(data_map->count(key))) {
                device_data = parafilter_mmr::mem_allocator(read_size);
                (*data_map)[key] = device_data;
            } else {
                device_data = (*data_map)[key];
            }

            cudaMemcpy(device_data, data, read_size, cudaMemcpyHostToDevice);
            free(data);
        }
    }
}

inline void generate_label_constrains(raft::device_resources const& dev_resources,
                               raft::device_matrix_view<float, uint64_t> ranges, 
                               raft::device_matrix_view<float, uint64_t> labels)
{
  uint64_t n_constrains = ranges.extent(0);
  uint64_t l_dim = ranges.extent(1) / 2;
  float* host_ranges = new float[n_constrains * 2 * l_dim];

  for (uint64_t i = 0; i < n_constrains; i++) {
    for (uint64_t j = 0; j < l_dim; j++) {
      float r = static_cast<float>(rand() % 10000) / static_cast<float>(10000);
      float l = static_cast<float>(rand() % 10000) / static_cast<float>(10000) - 1.f;

      std::cout << r << " " << l << " " << (r + l) / (r - l) << "\n";

      uint64_t cur_range = i * l_dim * 2 + j * 2;
      host_ranges[cur_range] = l;
      host_ranges[cur_range + 1] = r; 

      std::cout << cur_range << " " << cur_range + 1 << "\n";
    }
  }

  cudaMemcpy(ranges.data_handle(), host_ranges, 2 * l_dim * n_constrains * sizeof(float), cudaMemcpyHostToDevice);

  raft::random::RngState r(1234ULL);
  raft::random::uniform(dev_resources,
                        r,
                        raft::make_device_vector_view(labels.data_handle(), labels.size()),
                        -1.0f,
                        1.0f);
  
  delete [] host_ranges;
}

template<typename T, typename ElementType>
void print_raft_view(raft::device_resources const &dev_resources, 
                    T const& device_view) 
{
  size_t n_row = device_view.extent(0);
  size_t n_dim = device_view.extent(1);

  std::cout << "matrix view with rows: " << n_row << ", dimension: " << n_dim << "\n";

  // for 1 dimensional data, extent 1 is 0
  if (!n_dim) n_dim = 1;
  ElementType* host_ptr = new ElementType[n_row * n_dim];

  cudaMemcpy(host_ptr, device_view.data_handle(), n_row * n_dim * sizeof(ElementType), cudaMemcpyDeviceToHost);
  std::cout << "copy success!\n";

  for (int i = 0; i < n_row; i++) {
    std::cout << "[";
    for (int j = 0; j < n_dim; j++) {
      std::cout << host_ptr[i * n_dim + j];
      if (j != n_dim - 1)
        std::cout << ",";
    }
    std::cout << "]\n";
  }

  delete [] host_ptr;
}

template<typename T> void dbg_print_cuda_device_mem(T* p, int64_t size) 
{
    int n_ele = size / sizeof(T);
    T* host_ptr = new T[n_ele];

    cudaMemcpy(host_ptr, p, size, cudaMemcpyDeviceToHost);
    checkCUDAErrorWithLine("cuda error with cuda memcpy failed!");

    for (int i = 0; i < n_ele; i++)
        std::cout << host_ptr[i] << "\n";
}

#define INIT_PARAFILTER \
thread_local Timer global_timer; \
thread_local double query_time; \
thread_local double build_time; \
bool fake_run = false; \
bool parafilter_mmr::fake_run = false; \ 
bool break_down = false; \
bool run_ground_truth = false; \
const int SWAP_BUFF_COUNT = 2; \
std::vector<uint64_t> parafilter_mmr::total = {}; \
std::vector<uint64_t> parafilter_mmr::available = {}; \
std::vector<std::map<uint64_t, std::vector<void *>>> parafilter_mmr::cur_mems = {}; \
std::vector<std::map<uint64_t, int>> parafilter_mmr::cur_offset = {}; \
thread_local cudaStream_t dis_copy_stream = nullptr; \
thread_local cudaStream_t idx_copy_stream = nullptr; \
std::map<std::string, int> parafilter_config::str_to_offset_map = { \
      {"DATA_WIDTH", offsetof(parafilter_config, data_width)}, \
      {"INDEX_WIDTH", offsetof(parafilter_config, index_width)}, \
      {"DATASET", offsetof(parafilter_config, dataset)}, \
      {"PATH", offsetof(parafilter_config, path)}, \
      {"PQ_DIM", offsetof(parafilter_config, pq_dim)}, \
      {"N_CLUSTERS", offsetof(parafilter_config, n_clusters)}, \
      {"EXP1", offsetof(parafilter_config, exp1)}, \
      {"EXP2", offsetof(parafilter_config, exp2)}, \
      {"TOPK", offsetof(parafilter_config, topk)}, \
      {"BREAK_DOWN", offsetof(parafilter_config, break_down)}, \
      {"ENABLE_MULTI_GPU", offsetof(parafilter_config, enable_multi_gpu)}, \
      {"MEM_BOUND", offsetof(parafilter_config, mem_bound)}, \
      {"IS_CALC_MEM_PREDICTOR_COEFF", offsetof(parafilter_config, is_calc_mem_predictor_coeff)}, \
      {"MERGE_RATE", offsetof(parafilter_config, merge_rate)}, \
      {"LOWEST_QUERY_BATCH", offsetof(parafilter_config, lowest_query_batch)}, \
      {"FILTER_DIM", offsetof(parafilter_config, filter_dim)}, \
};