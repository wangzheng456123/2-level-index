#include <core/preprocessing.cuh>

__global__ void denormalize_labels_kernel(const float* data, uint64_t n_data,
    const float* shift_val, const int* map_types, const float* interval_map, const int* div_values, 
    uint64_t l, float* out)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_data) return;

    for (uint64_t i = 0; i < l; ++i) {
        uint64_t idx = tid * l + i;

        if (map_types[i] == 0) {
            out[idx * 2] = data[idx] - max(shift_val[2 * i], 1e-7);
            out[idx * 2 + 1] = data[idx] + max(shift_val[2 * i + 1], 1e-7);
        }
        else if (map_types[i] == 1) {
            int val = data[idx] - 1;
            out[idx * 2] = interval_map[2 * val];
            out[idx * 2 + 1] = interval_map[2 * val + 1];
        }
        else if (map_types[i] == 2) {
            int val = data[idx];
            out[idx * 2] = val / div_values[i];
            out[idx * 2 + 1] = std::numeric_limits<float>::max();
        }
        else if (map_types[i] == 3) {
            out[2 * idx] = data[2 * idx];
            out[2 * idx + 1] = data[2 * idx + 1];
        }
    }
}

__global__ void normalize_ranges_kernel(float* normalized_data_labels,
    uint64_t n_data, float* global_min, float* global_max, const float* ranges, uint64_t l)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_data) return;

    for (uint64_t i = 0; i < l; ++i) {
        uint64_t idx = tid * l + i;
        float left = ranges[idx * 2];
        float right = ranges[idx * 2 + 1];
        float midpoint = (left + right) / 2.0f;
        float coeff = 1.0f;
        // float coeff = 2.f / (right - left);
        // coeff = 2.f / (global_max[i] - global_min[i]);
        normalized_data_labels[idx] = (midpoint - global_min[i]) * coeff;
    }
}

__global__ void normalize_data_labels_kernel(const float* data, uint64_t n_data,
    float* global_max, uint64_t l, float* out)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_data) return;

    for (uint64_t i = 0; i < l; ++i) {
        uint64_t idx = tid * l + i;
        float coeff = 0.f;

        coeff = 2.f / global_max[i];
        out[idx] = data[idx] * coeff;
    }
}

void preprocessing_labels(
    raft::device_matrix_view<float, uint64_t> data_labels, 
    raft::device_matrix_view<float, uint64_t> normalized_data_labels,
    raft::device_matrix_view<float, uint64_t> query_labels, 
    raft::device_matrix_view<float, uint64_t> normalized_query_labels,
    raft::device_matrix_view<float, uint64_t> ranges,
    float* global_min_dev, 
    float* global_max_dev,
    const filter_config& f_config, 
    bool is_query_changed,
    bool is_data_changed, 
    bool reconfig
)
{
    uint64_t n_data = normalized_data_labels.extent(0);
    uint64_t n_queries = normalized_query_labels.extent(0);

    uint64_t l_dim = normalized_data_labels.extent(1);

    // todo fuse the 3 kernel calls to 1
    if (is_data_changed) {
        preprocessing_data_labels(data_labels, normalized_data_labels, global_max_dev);
    }

    if (is_query_changed) {
        recovery_filters(query_labels, ranges, f_config);
        normalize_ranges(ranges, normalized_query_labels, global_min_dev, global_max_dev);
    }
}

static void process_filter_config(
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

static void build_filter_device_filter_parameters(filter_parameters_dev_t &filter_parameters_dev, 
                                                  const filter_config& f_config
                                                  )
{
    if (filter_parameters_dev.is_configed) return;
    auto trans_vec_to_device = [](void*& dev_ptr, const void* src, size_t size) {
        dev_ptr = parafilter_mmr::mem_allocator(size);
        cudaMemcpy(dev_ptr, src, size, cudaMemcpyHostToDevice);
    };
    trans_vec_to_device(reinterpret_cast<void*&>(filter_parameters_dev.shift_val_dev), 
                f_config.shift_val.data(), f_config.shift_val.size() * sizeof(float));

    std::vector<float> ranges_map;
    for (const auto& maps : f_config.interval_map) {
        for (int value : maps) {
            ranges_map.push_back(static_cast<float>(value));
        }
    }
    trans_vec_to_device(reinterpret_cast<void*&>(filter_parameters_dev.ranges_map_dev), 
                ranges_map.data(), ranges_map.size() * sizeof(float));

    std::vector<float> shift_len;
    std::vector<std::vector<float>> maps_len;
    process_filter_config(f_config, shift_len, maps_len);

    trans_vec_to_device(reinterpret_cast<void*&>(filter_parameters_dev.shift_len_dev), 
                shift_len.data(), shift_len.size() * sizeof(float));

    std::vector<float> maps_len_flat;
    for (const auto& map : maps_len) {
        maps_len_flat.insert(maps_len_flat.end(), map.begin(), map.end());
    }
    trans_vec_to_device(reinterpret_cast<void*&>(filter_parameters_dev.maps_len_dev), 
                            maps_len_flat.data(), maps_len_flat.size() * sizeof(float));
    trans_vec_to_device(reinterpret_cast<void*&>(filter_parameters_dev.map_types_dev), 
                            f_config.filter_type.data(), f_config.filter_type.size() * sizeof(int));
    trans_vec_to_device(reinterpret_cast<void*&>(filter_parameters_dev.div_value_dev), 
                            f_config.div_value.data(), f_config.div_value.size() * sizeof(int));

    filter_parameters_dev.is_configed = true;
}

void calculate_batch_min_max(raft::device_matrix_view<float, uint64_t> const& data_labels,
                             std::vector<float> &global_min, std::vector<float> &global_max,
                             uint64_t l 
                             ) 
{
    for (int i = 0; i < l; i++) {
      global_min[i] = std::numeric_limits<float>::max();
      global_max[i] = std::numeric_limits<float>::lowest();
    }

    uint64_t n_data = data_labels.extent(0);
    uint64_t l_data = data_labels.extent(1);

    float* data_transposed_device = nullptr;
    cudaMalloc(&data_transposed_device, n_data * l_data * sizeof(float));
    checkCUDAErrorWithLine("CUDA malloc for data_transposed_device failed.");
    transpose_matrix(data_labels.data_handle(), data_transposed_device, n_data, l_data);

    std::vector<float> row_min_host(l);
    std::vector<float> row_max_host(l);

    uint64_t row_len = n_data * (l_data / l);
    for (int i = 0; i < l; i++) {
        global_min[i] = array_min_max_reduce(data_transposed_device + i * row_len, row_len, true);
        global_max[i] = array_min_max_reduce(data_transposed_device + i * row_len, row_len, false);
    }

    cudaFree(data_transposed_device);
    checkCUDAErrorWithLine("CUDA free failed.");
}

void recovery_filters(
    raft::device_matrix_view<float, uint64_t> const& query_labels,
    raft::device_matrix_view<float, uint64_t> &ranges, 
    const filter_config& f_config
)
{
    uint64_t n_queries = query_labels.extent(0);

    thread_local filter_parameters_dev_t filter_parameters_dev;
    build_filter_device_filter_parameters(filter_parameters_dev, f_config);  

    dim3 full_block_per_grid((n_queries + block_size - 1) / block_size);
    // Call denormalize_labels_kernel
    denormalize_labels_kernel<<<full_block_per_grid, block_size>>>(
        query_labels.data_handle(),           // Input raw query labels
        n_queries,                                // Number of queries
        filter_parameters_dev.shift_val_dev,  // Device pointer for shift values
        filter_parameters_dev.map_types_dev,  // Device pointer for map types
        filter_parameters_dev.ranges_map_dev, // Device pointer for interval map
        filter_parameters_dev.div_value_dev,  // Device pointer for div value
        f_config.l,                           // Length of intervals
        ranges.data_handle() // Output to denormalized_query_labels
    );

}

void normalize_ranges(
    raft::device_matrix_view<float, uint64_t> ranges, 
    raft::device_matrix_view<float, uint64_t> normalized_ranges, 
    float* global_min_dev, 
    float* global_max_dev
)
{
    uint64_t n_row = ranges.extent(0);
    uint64_t n_dim = ranges.extent(1);
    uint64_t l = n_dim / 2;

    dim3 full_block_per_grid((n_row + block_size - 1) / block_size);

    // Call normalize_ranges_labels_kernel to replace normalize_labels_kernel
    normalize_ranges_kernel<<<full_block_per_grid, block_size>>>(
        normalized_ranges.data_handle(),           // Output to normalized_data_labels
        n_row,                                           // Number of data points
        global_min_dev,                                  // Global minimum value
        global_max_dev,                                  // Global maximum value
        ranges.data_handle(),         // Device pointer for ranges
        l                                       // Length of intervals
    ); 
}

void preprocessing_data_labels(raft::device_matrix_view<float, uint64_t> data_labels,   
                               raft::device_matrix_view<float, uint64_t> normalized_data_labels, 
                               float* global_max_dev)
{
    uint64_t n_data = data_labels.extent(0);
    uint64_t l = data_labels.extent(1);

    dim3 threads_per_block(64, 4);
    dim3 blocks_per_grid((n_data + threads_per_block.x - 1) / threads_per_block.x, (l + threads_per_block.y) + threads_per_block.y);

    normalize_data_labels_kernel<<<threads_per_block, blocks_per_grid>>>(data_labels.data_handle(), 
            n_data, global_max_dev, l, normalized_data_labels.data_handle());
}