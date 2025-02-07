/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "para_filter.cuh"
#include "verification.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/operators.hpp>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/refine.cuh>
#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/matrix/slice.cuh>
#include <raft/cluster/kmeans.cuh>
#include <raft/cluster/kmeans_types.hpp>
#include <raft/matrix/select_k.cuh>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <cstdint>
#include <thread>

/*
  this macro defines and initializes some useful globla variable
  and static class members. it should be include in your source file
  once and only once
*/
void get_parafilter_config(parafilter_config **conf, const char path[] = "parafilter.conf") {
  *conf = new parafilter_config(path);
}

void parafilter_build(raft::device_resources const& dev_resources, // in: the raft resources
                      raft::device_matrix_view<const float, uint64_t> dataset, //in: the training dataset
                      size_t pq_dim,  //in: dimesnion of the pq vector
                      size_t pq_len, 
                      size_t n_clusters, //in: number of clusters for each subspace
                      raft::device_matrix_view<uint8_t, uint64_t> codebook, //out: codebook
                      raft::device_matrix_view<float, uint64_t> centers) // out: centers
{
  uint64_t n_row = dataset.extent(0);
  uint64_t n_dim = dataset.extent(1);

  std::cout << "build pq codebook with dataset size: " << n_row << ", data dimension: " << n_dim << 
    ", pq dimension: " << pq_dim << ", number of cluserters: " << n_clusters << "\n";

  // todo: properly processing n_dim is'n multiple of pq_dim
  auto tmp_train = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_row, pq_len);
  auto tmp_labels = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(n_row);

  // use 2 dimesnional matrix as the low level container
  uint64_t centers_len = n_clusters * pq_len;

  for (int i = 0; i < pq_dim; i++) {
    if (i == pq_dim - 1) {
      cudaMemset(tmp_train.data_handle(), 0, tmp_train.size());
    }
    slice_coordinates<uint64_t> coords{0, i * pq_len, n_row, std::min((i + 1) * pq_len, n_dim)};
    // todo: remove this copy since it is time consuming for large dataset.
    slice(dev_resources, dataset, tmp_train, coords);

    raft::cluster::kmeans::KMeansParams params;
    params.n_clusters = n_clusters;

    float interia;
    uint64_t niters;

    auto cur_centers_view = raft::make_device_matrix_view<float, uint64_t>(centers.data_handle() + centers_len * i, n_clusters, pq_len);

    raft::cluster::kmeans::fit_predict<float, uint64_t>(
                                       dev_resources, 
                                       params, 
                                       tmp_train, 
                                       std::nullopt, 
                                       cur_centers_view, 
                                       tmp_labels,
                                       raft::make_host_scalar_view(&interia), 
                                       raft::make_host_scalar_view(&niters));

    auto tmp_quanted_vector_view = raft::make_device_vector_view<uint8_t, uint64_t>(codebook.data_handle() + i * n_row, n_row);

    raft::linalg::map_offset(dev_resources,  
                             tmp_quanted_vector_view, 
                             [] __device__ (const uint64_t idx, const uint64_t ele) {
                                return static_cast<uint8_t>(ele);
                             },
                             raft::make_const_mdspan(tmp_labels));
  }

  std::cout << "para-filter index build success\n";
}

void parafilter_query(raft::device_resources const& dev_resources,
                      raft::device_matrix_view<uint8_t, uint64_t> const& codebook,
                      raft::device_matrix_view<float, uint64_t> const& dataset,
                      raft::device_matrix_view<float, uint64_t> const& centers,
                      raft::device_matrix_view<float, uint64_t> const& queries,
                      raft::device_matrix_view<float, uint64_t> const& data_labels,
                      raft::device_matrix_view<float, uint64_t> const& normalized_data_labels,
                      raft::device_matrix_view<float, uint64_t> const& query_labels,
                      raft::device_matrix_view<float, uint64_t> const& normalized_query_labels,
                      raft::device_matrix_view<float, uint64_t> const& ranges,
                      raft::device_matrix_view<float, uint64_t> selected_distance, 
                      raft::device_matrix_view<uint64_t, uint64_t> selected_indices,
                      uint32_t* exps,
                      int pq_dim, 
                      int pq_len, 
                      int topk, 
                      float merge_rate) 
{    
  int n_data = dataset.extent(0);
  int n_queries = queries.extent(0);
  int n_dim = dataset.extent(1);
  int l = data_labels.extent(1);

  auto vec_dis = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, n_data); 
  // todo: batch size needs modification
  calc_batched_L2_distance(dev_resources, queries, codebook, centers, vec_dis, pq_dim, pq_len);

  auto label_dis = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, n_data); 

  // calculated distance between data lables and normalized constrains
  auto metric = raft::distance::DistanceType::L2Expanded;

  raft::distance::pairwise_distance(dev_resources, 
          normalized_query_labels, normalized_data_labels, label_dis, metric);

  auto label_dis_ptr = label_dis.data_handle();
  // sum distance of vectors and lebels
  auto dis = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, n_data);
  matrix_add_with_weights<float, uint64_t>(dev_resources, vec_dis, label_dis, dis, 1.f, merge_rate);
  
  auto first_candi_labels = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, topk * exps[0] * l);
  auto first_val = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, topk * exps[0]);
  auto first_idx = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, topk * exps[0]);

  raft::matrix::select_k<float, uint64_t>(dev_resources, dis, std::nullopt, first_val, first_idx, true, true);
  select_elements<float, uint64_t>(dev_resources, data_labels, first_idx, first_candi_labels);
  select_elements<float, uint64_t>(dev_resources, vec_dis, first_idx, first_val, false);

  cudaDeviceSynchronize();
  compute_matches_on_cpu(first_idx.data_handle(), first_idx.extent(0), first_idx.extent(1), );

  auto second_indices = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, topk * exps[1]);
  filter_candi_by_labels(dev_resources, first_candi_labels, ranges, first_val, topk * exps[1], second_indices);

  auto second_indices_direct = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, topk * exps[1]);
  select_elements<uint64_t, uint64_t>(dev_resources, first_idx, second_indices, second_indices_direct, false);
  auto second_indices_direct_ptr = second_indices_direct.data_handle();

  refine<float, uint64_t>(
         dev_resources, 
         dataset,
         queries,  
         second_indices_direct, 
         selected_indices,
         selected_distance);
} 

// todo: can this abstraction encapsulation a class?
inline void parafilter_calc_input_size_map(std::map<std::string, std::pair<uint64_t, uint64_t>> &input_size_map, 
                                    size_t pq_dim, 
                                    size_t n_dim, 
                                    size_t l, 
                                    size_t n_data, 
                                    size_t n_queries, 
                                    size_t n_clusters, 
                                    size_t topk) 
{
    size_t pq_len = (n_dim + pq_dim - 1) / pq_dim;

    input_size_map["dataset"] = {n_data, n_dim};
    input_size_map["queries"] = {n_queries, n_dim};
    input_size_map["codebook"] = {pq_dim, n_data};
    input_size_map["centers"] = {pq_dim, pq_len * n_clusters};
    input_size_map["data_labels"] = {n_data, l};
    input_size_map["query_labels"] = {n_queries, l};
    input_size_map["ranges"] = {n_queries, 2 * l};
    input_size_map["selected_indices"] = {n_queries, 2 * topk};
    input_size_map["selected_distance"] = {n_queries, 2 * topk};
}

// todo1: generalize it to any procedures
// todo2: not allocate actual cuda memory in fake run 
void parafilter_build_run(raft::device_resources const& dev_resources,
                          raft::device_matrix_view<float, uint64_t> const& queries,
                          raft::device_matrix_view<float, uint64_t> const& dataset,
                          raft::device_matrix_view<float, uint64_t> const& data_labels,
                          raft::device_matrix_view<float, uint64_t> const& query_labels,
                          raft::device_matrix_view<float, uint64_t> selected_distance,
                          raft::device_matrix_view<uint64_t, uint64_t> selected_indices,
                          size_t pq_dim,
                          size_t n_clusters,
                          uint32_t* exps,
                          int topk,
                          float* global_min,
                          float* global_max,
                          const filter_config &f_config, 
                          float merge_rate = 0.035, 
                          bool run_build = true, 
                          bool reconfig = false) 
{
    size_t n_data = dataset.extent(0);
    size_t n_queries = queries.extent(0);
    size_t n_dim = dataset.extent(1);
    size_t pq_len = (n_dim + pq_dim - 1) / pq_dim;
    size_t l = data_labels.extent(1);

    thread_local raft::device_matrix_view<uint8_t, uint64_t> codebook{};
    thread_local raft::device_matrix_view<float, uint64_t> centers{};
    if (run_build) {
        codebook = parafilter_mmr::make_device_matrix_view<uint8_t, uint64_t>(pq_dim, n_data);
        centers = parafilter_mmr::make_device_matrix_view<float, uint64_t>(pq_dim, n_clusters * pq_len);
        parafilterPerfLogWraper(parafilter_build(dev_resources, dataset, pq_dim, pq_len, n_clusters, codebook, centers), build_time);
    }

    bool is_data_changed = false;
    if (run_build) is_data_changed = true;
    auto normalized_query_labels = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, l);
    auto ranges = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, 2 * l);

    thread_local raft::device_matrix_view<float, uint64_t> normalized_data_labels{};
    if (is_data_changed) { 
      normalized_data_labels = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_data, l);
    }

    parafilterPerfLogWraper(
              (preprocessing_labels<float, uint64_t>)(
                    dev_resources, 
                    data_labels, 
                    normalized_data_labels, 
                    query_labels, 
                    normalized_query_labels, 
                    ranges, 
                    global_min, 
                    global_max, 
                    f_config, 
                    true, 
                    is_data_changed, 
                    reconfig
                ), 
                    query_time
    );

    parafilterPerfLogWraper(
              parafilter_query(dev_resources, 
                     codebook, 
                     dataset, 
                     centers, 
                     queries, 
                     data_labels, 
                     normalized_data_labels,
                     query_labels,
                     normalized_query_labels, 
                     ranges, 
                     selected_distance, 
                     selected_indices, 
                     exps, 
                     pq_dim, 
                     pq_len, 
                     topk, 
                     merge_rate
                  ), 
                     query_time
    );
}

void calc_mem_predictor_coeff(raft::device_resources const& dev_resources, 
                              size_t pq_dim,
                              size_t n_dim, 
                              size_t l, 
                              size_t n_data, 
                              size_t n_queries, 
                              size_t n_clusters, 
                              uint32_t* exps,
                              int topk, 
                              double* coeff, 
                              const filter_config &f_config) 
{
  // run fake ness 4 times, to solve the linear equation :
  // size = a * n_data + b * n_queries + c * n_data * n_queries + d
  std::map<std::string, std::pair<uint64_t, uint64_t>> input_size_map;

  // todo: move it to a function or class
  #define build_fake_run_input \
  queries_fake = parafilter_mmr::make_device_matrix_view<float, uint64_t>(input_size_map["queries"].first, input_size_map["queries"].second); \
  dataset_fake = parafilter_mmr::make_device_matrix_view<float, uint64_t>(input_size_map["dataset"].first, input_size_map["dataset"].second); \
  codebook_fake = parafilter_mmr::make_device_matrix_view<uint8_t, uint64_t>(input_size_map["codebook"].first, input_size_map["codebook"].second); \
  centers_fake = parafilter_mmr::make_device_matrix_view<float, uint64_t>(input_size_map["centers"].first, input_size_map["centers"].second); \
  data_labels_fake = parafilter_mmr::make_device_matrix_view<float, uint64_t>(input_size_map["data_labels"].first, input_size_map["data_labels"].second); \
  query_labels_fake = parafilter_mmr::make_device_matrix_view<float, uint64_t>(input_size_map["query_labels"].first, input_size_map["query_labels"].second); \
  selected_indices_fake = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(input_size_map["selected_indices"].first, input_size_map["selected_indices"].second); \
  selected_distance_fake = parafilter_mmr::make_device_matrix_view<float, uint64_t>(input_size_map["selected_distance"].first, input_size_map["selected_distance"].second);

  #define define_raft_input \ 
  raft::device_matrix_view<float, uint64_t> queries_fake{}; \
  raft::device_matrix_view<float, uint64_t> dataset_fake{}; \
  raft::device_matrix_view<uint8_t, uint64_t> codebook_fake{}; \
  raft::device_matrix_view<float, uint64_t> centers_fake{}; \
  raft::device_matrix_view<float, uint64_t> data_labels_fake{}; \
  raft::device_matrix_view<float, uint64_t> query_labels_fake{}; \
  raft::device_matrix_view<float, uint64_t> ranges_fake{}; \
  raft::device_matrix_view<uint64_t, uint64_t> selected_indices_fake{}; \
  raft::device_matrix_view<float, uint64_t> selected_distance_fake{}; 

  std::vector<float> null_global(l, 0);

  #define call_fake_run \
    parafilter_build_run(dev_resources, queries_fake, dataset_fake, \ 
                          data_labels_fake, query_labels_fake, selected_distance_fake, \ 
                          selected_indices_fake, pq_dim, n_clusters, exps, topk, null_global.data(), null_global.data(), f_config, 0.035, true, true); 
  
  #define call_calc_fake_matrix_size \
      parafilter_calc_input_size_map(input_size_map, pq_dim, n_dim, l, fake_data_cnt, fake_query_cnt, n_clusters, topk);

  // todo: binding these to a function or class
  double mat[4][5];
  auto write_mat = [&mat] (int row, size_t fake_data, size_t fake_query, uint64_t mem_used) {
    mat[row][0] = fake_data;
    mat[row][1] = fake_query;
    mat[row][2] = fake_data * fake_query;
    mat[row][3] = 1;
    mat[row][4] = mem_used;
  };

  // todo mergency: allocated tmp fake matrix needs to be deallocated
  define_raft_input
  size_t fake_data_cnt = topk * exps[0] + 10;
  size_t fake_query_cnt = 5;
  call_calc_fake_matrix_size
  build_fake_run_input
  call_fake_run
  uint64_t used1 = parafilter_mmr::get_current_workspace_used();
  write_mat(0, fake_data_cnt, fake_query_cnt, used1);
  parafilter_mmr::free_cur_workspace_device_mems();
  parafilter_mmr::reset_current_workspace(2ull * 1024 * 1024 * 1024);

  fake_data_cnt = topk * exps[0] + 100;
  fake_query_cnt = 3;
  call_calc_fake_matrix_size
  build_fake_run_input
  call_fake_run
  uint64_t used2 = parafilter_mmr::get_current_workspace_used();
  write_mat(1, fake_data_cnt, fake_query_cnt, used2);
  parafilter_mmr::free_cur_workspace_device_mems();
  parafilter_mmr::reset_current_workspace(2ull * 1024 * 1024 * 1024);

  fake_data_cnt = topk * exps[0] + 1000;
  fake_query_cnt = 2;
  call_calc_fake_matrix_size
  build_fake_run_input
  call_fake_run
  uint64_t used3 = parafilter_mmr::get_current_workspace_used();
  write_mat(2, fake_data_cnt, fake_query_cnt, used3);
  parafilter_mmr::free_cur_workspace_device_mems();
  parafilter_mmr::reset_current_workspace(2ull * 1024 * 1024 * 1024);

  fake_data_cnt = topk * 2 * exps[0];
  fake_query_cnt = 1;
  call_calc_fake_matrix_size
  build_fake_run_input
  call_fake_run
  uint64_t used4 = parafilter_mmr::get_current_workspace_used();
  write_mat(3, fake_data_cnt, fake_query_cnt, used4);
  parafilter_mmr::free_cur_workspace_device_mems();
  parafilter_mmr::reset_current_workspace(2ull * 1024 * 1024 * 1024);

  LOG(INFO) << used1 << " " << used2 << " " << used3 << " " << used4 << "\n";

  gauss(mat, 4);
  for (int i = 0; i < 4; i++)
    coeff[i] = mat[i][4];
}

void split_task(double* coeff,
                uint64_t n_data, 
                uint64_t n_queries,
                int topk,
                uint64_t mem_bound, 
                uint64_t &query_batch_size, 
                uint64_t &data_batch_size, 
                uint64_t aditional = 0, 
                uint64_t lowest_query_batch_size = 125)  
{
  uint64_t available, total;
  get_current_device_mem_info(available, total);
  int id;
  cudaGetDevice(&id);
  LOG(INFO) << "device: " << id << "available: " << available << " upper bound: "
    << mem_bound;

  uint64_t upper_bound = std::min(available / 5, mem_bound);

  auto bisearch_proper_split = [coeff, n_data, topk, n_queries, &query_batch_size, &data_batch_size] (uint64_t upper_bound) {
    uint64_t r_d = n_data;
    uint64_t l_d = topk;

    uint64_t r_q = n_queries;
    uint64_t l_q = 1;

    while (1) {
      uint64_t mid_d = (r_d + l_d + 1) >> 1;

      r_q = n_queries;
      l_q = 1;

      while (l_q < r_q) {
        uint64_t mid_q = (l_q + r_q + 1) >> 1;
        uint64_t value = coeff[0] * mid_d + 
                         coeff[1] * mid_q +
                         coeff[2] * mid_d * mid_q + 
                         coeff[3] + 1;

        if (value > upper_bound) r_q = mid_q - 1;
        else l_q = mid_q; 
      } 
      if (l_q < 125) r_d = mid_d - 1;
      else l_d = mid_d;
      if (l_d == r_d && l_q >= 125) break;
    }

    query_batch_size = l_q;
    data_batch_size = l_d;
  };
  // todo: when cannot find proper batch size for current upper bound, enlarge it
  bisearch_proper_split(upper_bound);
  while (n_queries % query_batch_size != 0) query_batch_size--;
  data_batch_size = findMaxFactor(data_batch_size, n_data);
}

void calculate_batch_min_max(const std::string& path, std::vector<float> &global_min, std::vector<float> &global_max, 
                             size_t l, size_t n_data, size_t n_queries) 
{
    size_t data_offset = 0, query_offset = 0;

    std::string train_file_path = path + "train_label";
    std::string test_file_path = path + "test_label";

    for (int i = 0; i < l; i++) {
      global_min[i] = std::numeric_limits<float>::max();
      global_max[i] = std::numeric_limits<float>::lowest();
    }

    uint64_t data_batch_size = 0, query_batch_size = 0;
    double coeff[4];
    coeff[0] = sizeof(float) * 2 * l;
    coeff[1] = sizeof(float) * 2 * l;
    coeff[2] = 0;
    coeff[3] = 0;

    split_task(coeff, n_data, n_queries, 1, 
              2ll * 1024 * 1024 * 1024, query_batch_size, data_batch_size);
    
    if (data_batch_size > 1e6) {
      data_batch_size = findMaxFactor(1e6, n_data);
    }

    while (data_offset < n_data * l * sizeof(float)) {
        size_t bytes_to_read = std::min(data_batch_size * l * sizeof(float), n_data * l * sizeof(float) - data_offset);
        void* data_host = read_binary_file(train_file_path, data_offset, bytes_to_read);

        size_t batch_elements = bytes_to_read / sizeof(float);
        size_t rows = data_batch_size;
        size_t cols = l;

        float* data_device = nullptr;
        cudaMalloc(&data_device, batch_elements * sizeof(float));
        checkCUDAErrorWithLine("CUDA malloc for data_device failed.");
        cudaMemcpy(data_device, data_host, batch_elements * sizeof(float), cudaMemcpyHostToDevice);
        checkCUDAErrorWithLine("CUDA memcpy to device failed.");

        float* data_transposed_device = nullptr;
        cudaMalloc(&data_transposed_device, batch_elements * sizeof(float));
        checkCUDAErrorWithLine("CUDA malloc for data_transposed_device failed.");
        transpose_matrix(data_device, data_transposed_device, rows, cols);

        std::vector<float> row_min_host(l);
        std::vector<float> row_max_host(l);

        for (int i = 0; i < l; i++) {
          row_min_host[i] = array_min_max_reduce(data_transposed_device + i * data_batch_size, query_batch_size, true);
          row_max_host[i] = array_min_max_reduce(data_transposed_device + i * data_batch_size, query_batch_size, false);
        }

        for (size_t i = 0; i < l; ++i) {
            global_min[i] = std::min(global_min[i], row_min_host[i]);
            global_max[i] = std::max(global_max[i], row_max_host[i]);
        }

        cudaFree(data_device);
        cudaFree(data_transposed_device);
        checkCUDAErrorWithLine("CUDA free failed.");

        free(data_host);
        data_offset += data_batch_size * l * sizeof(float);
    }

    while (query_offset < n_queries * l * sizeof(float)) {
        size_t bytes_to_read = std::min(query_batch_size * l * sizeof(float), n_queries * l * sizeof(float) - query_offset);
        void* data_host = read_binary_file(test_file_path, query_offset, bytes_to_read);

        size_t batch_elements = bytes_to_read / sizeof(float);
        size_t rows = query_batch_size;
        size_t cols = l;

        float* data_device = nullptr;
        cudaMalloc(&data_device, batch_elements * sizeof(float));
        checkCUDAErrorWithLine("CUDA malloc for data_device failed.");
        cudaMemcpy(data_device, data_host, batch_elements * sizeof(float), cudaMemcpyHostToDevice);
        checkCUDAErrorWithLine("CUDA memcpy to device failed.");

        float* data_transposed_device = nullptr;
        cudaMalloc(&data_transposed_device, batch_elements * sizeof(float));
        checkCUDAErrorWithLine("CUDA malloc for data_transposed_device failed.");
        transpose_matrix(data_device, data_transposed_device, rows, cols);

        std::vector<float> row_min_host(l);
        std::vector<float> row_max_host(l);
        for (int i = 0; i < l; i++) {
          row_min_host[i] = array_min_max_reduce(data_transposed_device + i * query_batch_size, query_batch_size, true);
          row_max_host[i] = array_min_max_reduce(data_transposed_device + i * query_batch_size, query_batch_size, false);
        }

        for (size_t i = 0; i < l; ++i) {
            global_min[i] = std::min(global_min[i], row_min_host[i]);
            global_max[i] = std::max(global_max[i], row_max_host[i]);
        }

        cudaFree(data_device);
        cudaFree(data_transposed_device);
        checkCUDAErrorWithLine("CUDA free failed.");

        free(data_host);
        query_offset += query_batch_size * l * sizeof(float);
    }
}

inline void split_uniform(uint64_t n, uint64_t m, uint64_t i, 
                         uint64_t &size, uint64_t &offset) 
{
  uint64_t r = n % m;
  if (i < r) {
    size = (n + m - 1) / m;
    offset = i * size;
  }
  else {
    size = n / m;
    offset = ((n + m - 1) / m) * r + (i - r) * size; 
  }
}

// todo: add it as a app configration.
#define MAX_TMP_RES_BUFF_SIZE 256 * 1024 * 1024
template<typename ElementType, typename IndexType>
void flush_current_res(ElementType *dis, IndexType* idx, size_t size, cudaEvent_t &dis_copy_done_event, 
                       cudaEvent_t &idx_copy_done_event, const cudaEvent_t &compute_done_event, int device_id, std::string const& path = "res/",
                       bool force_flush = false, bool overwrite = false, bool reset_offset = false) 
{
    thread_local ElementType* dis_buff = nullptr;
    thread_local IndexType* indices_buff = nullptr;
    thread_local size_t offset = 0;
    thread_local size_t buff_size = MAX_TMP_RES_BUFF_SIZE;
    thread_local size_t file_offset = 0;
    thread_local int thread_id = -1;
    thread_local cudaStream_t dis_copy_stream = nullptr;
    thread_local cudaStream_t idx_copy_stream = nullptr;

    if (reset_offset) {
      offset = 0;
      file_offset = 0;
    }

    if (thread_id == -1) {
        // Use a stable hashing method to ensure consistent thread ID
        thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 10000;
    }

    // Generate file paths based on device_id to ensure dictionary order and consistency
    std::string dis_file_path = path + "distances_" + std::to_string(device_id);
    std::string neigh_file_path = path + "neighbors_" + std::to_string(device_id);

    // Initialize CUDA streams if not already created
    if (dis_copy_stream == nullptr) {
        cudaStreamCreate(&dis_copy_stream);
    }
    if (idx_copy_stream == nullptr) {
        cudaStreamCreate(&idx_copy_stream);
    }

    // Allocate buffers if not already allocated
    if (dis_buff == nullptr) {
        dis_buff = new ElementType[buff_size];
    }
    if (indices_buff == nullptr) {
        indices_buff = new IndexType[buff_size];
    }

    // If buffer is full or forced to flush, write to disk
    if (offset + size >= buff_size || force_flush) {

        LOG(INFO) << "device: " << device_id << ", flush tmp res with offset: " << 
                offset << "with file offset: " << file_offset << ", overwrite: " << overwrite;
        auto dis_write_future = 
            write_binary_file_async(dis_file_path, file_offset * sizeof(ElementType), (void*)dis_buff, offset * sizeof(ElementType), !overwrite);
        auto neigh_write_future = 
            write_binary_file_async(neigh_file_path, file_offset * sizeof(IndexType), (void*)indices_buff, offset * sizeof(IndexType), !overwrite);

        if (force_flush) {
            // Wait for async writes to complete
            bool res = dis_write_future.get() && neigh_write_future.get();
            if (!res) {
                throw std::runtime_error("Failed to flush query result!");
            }
        }
        file_offset += offset;
        offset = 0; // Reset offset after flush
    }

    // If there is new data to process, copy it to the buffers asynchronously
    if (dis != nullptr && idx != nullptr) {
        assert(dis_copy_done_event != nullptr);
        assert(idx_copy_done_event != nullptr);
        assert(compute_done_event != nullptr);

        cudaStreamWaitEvent(dis_copy_stream, compute_done_event, 0);
        checkCUDAErrorWithLine("Insert wait for compute stream failed!");
        cudaMemcpyAsync(dis_buff + offset, dis, sizeof(ElementType) * size, cudaMemcpyDeviceToHost, dis_copy_stream);
        checkCUDAErrorWithLine("Copy temporary buffer to host failed!");
        cudaEventRecord(dis_copy_done_event, dis_copy_stream);

        cudaStreamWaitEvent(idx_copy_stream, compute_done_event, 0);
        cudaMemcpyAsync(indices_buff + offset, idx, sizeof(IndexType) * size, cudaMemcpyDeviceToHost, idx_copy_stream);
        checkCUDAErrorWithLine("Copy temporary buffer to host failed!");
        cudaEventRecord(idx_copy_done_event, idx_copy_stream);
    
        offset += size;
    }
}

bool write_coeff_to_binary(const std::string& file_path, const double coeff[4]) {
    std::ofstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << file_path << std::endl;
        return false;
    }

    file.write(reinterpret_cast<const char*>(coeff), sizeof(double) * 4);
    if (!file) {
        std::cerr << "Failed to write data to file: " << file_path << std::endl;
        return false;
    }
    file.close();
    return true;
}

bool read_coeff_from_binary(const std::string& file_path, double coeff[4]) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for reading: " << file_path << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(coeff), sizeof(double) * 4);
    if (!file) {
        std::cerr << "Failed to read data from file: " << file_path << std::endl;
        return false;
    }
    file.close();
    return true;
}

int main()
{
  parafilter_mmr::init_mmr();

  parafilter_config *p_config = nullptr;

  get_parafilter_config(&p_config);
  std::string dataset_path = p_config->path;

  std::vector<std::string> keys;
  get_data_set_list(keys, std::string(dataset_path + "keys").c_str());

  std::map<std::string, std::string> types;
  get_data_type_list(types, std::string(dataset_path + "dtypes").c_str(), keys);

  std::string filter_conf_path = dataset_path + "filter.conf";

  filter_config f_config(filter_conf_path);

  if (p_config->break_down) break_down = true;
  std::map<std::string, std::pair<int32_t, int32_t>> size_map;

  build_dataset(keys, types, nullptr, size_map, dataset_path);

  int64_t tot_samples = size_map["train_vec"].first;
  int64_t n_dim     = size_map["train_vec"].second;
  int64_t tot_queries = size_map["test_vec"].first;
  uint64_t l = size_map["train_label"].second;

  uint64_t pq_dim = p_config->pq_dim;
  uint64_t n_clusters = p_config->n_clusters;

  uint32_t exps[2];
  exps[0] = p_config->exp1;
  exps[1] = p_config->exp2;
  uint32_t topk = p_config->topk;
  float merge_rate = p_config->merge_rate;

  double coeff[4];
  if (p_config->is_calc_mem_predictor_coeff) {
    raft::device_resources dev_resources;
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1 * 1024 * 1024 * 1024ull);
    rmm::mr::set_current_device_resource(&pool_mr);
    
    calc_mem_predictor_coeff(dev_resources, pq_dim, n_dim, l, tot_samples, 
            tot_queries, n_clusters, exps, topk, coeff, f_config);
    write_coeff_to_binary("coeff", coeff);
    return 0;
  }
  read_coeff_from_binary("coeff", coeff);

  std::vector<float> global_min(l), global_max(l);
  calculate_batch_min_max(p_config->path, global_min, global_max, l, tot_samples, tot_queries);
  
  int device_count;
  cudaGetDeviceCount(&device_count);
  if (!p_config->enable_multi_gpu) device_count = 1;

  auto per_device_worker = [&](uint32_t i) {
    cudaSetDevice(i);

    // fixme: avoid to use raft mem pool
    raft::device_resources dev_resources;
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1 * 1024 * 1024 * 1024ull);
    rmm::mr::set_current_device_resource(&pool_mr);

    uint64_t query_offset, n_queries;
    split_uniform(tot_queries, device_count, i, n_queries, query_offset);
    uint64_t query_batch_size, data_batch_size;
    split_task(coeff, tot_samples, n_queries, topk, p_config->mem_bound, 
        query_batch_size, data_batch_size);

    LOG(INFO) << "device :" << i << " choose data batach size:" 
        << data_batch_size << " query batch size" <<  query_batch_size;

    uint64_t inter_buffer_size = topk * query_batch_size;
    // todo: ping pang the output buffer to increase throughput
    float* selected_distance_device_ptr;
    cudaMalloc((void **)&selected_distance_device_ptr, SWAP_BUFF_COUNT * inter_buffer_size * sizeof(float));
    uint64_t* selected_indices_device_ptr;
    cudaMalloc((void **)&selected_indices_device_ptr, SWAP_BUFF_COUNT * inter_buffer_size * sizeof(uint64_t));

    raft::device_matrix_view<float, uint64_t> selected_distance{};
    raft::device_matrix_view<uint64_t, uint64_t> selected_indices{};
    
    std::vector<cudaEvent_t> dis_copy_done_event(SWAP_BUFF_COUNT);
    std::vector<cudaEvent_t> idx_copy_done_event(SWAP_BUFF_COUNT);
    std::vector<cudaEvent_t> compute_done_event(SWAP_BUFF_COUNT);

    for (int buff_id = 0; buff_id < SWAP_BUFF_COUNT; buff_id++) {
      cudaEventCreate(&dis_copy_done_event[buff_id]);
      cudaEventCreate(&idx_copy_done_event[buff_id]);
      cudaEventCreate(&compute_done_event[buff_id]);
      checkCUDAErrorWithLine("failed to create flush event");
    }
    std::map<std::string, void*> data_map;
    int cur_res_buff_offset = 0;
    for (uint64_t data_batch_offset = 0; data_batch_offset < tot_samples; data_batch_offset += data_batch_size) {
      uint64_t cur_data_batch_size;
      cur_data_batch_size = data_batch_size;

      for (uint64_t query_batch_offset = 0; query_batch_offset < n_queries; query_batch_offset += query_batch_size) {        
        uint64_t cur_query_batch_size = query_batch_size;
        uint64_t cur_query_offset = query_offset + query_batch_offset;

        build_dataset(keys, types, &data_map, size_map, dataset_path, data_batch_offset, cur_data_batch_size, cur_query_offset, cur_query_batch_size);
        auto dataset      = raft::make_device_matrix_view<float, uint64_t>((float *)data_map["train_vec"], cur_data_batch_size, n_dim);
        auto queries      = raft::make_device_matrix_view<float, uint64_t>((float *)data_map["test_vec"], cur_query_batch_size, n_dim);

        auto data_labels = raft::make_device_matrix_view<float, uint64_t>((float *)data_map["train_label"], cur_data_batch_size, l);
        auto query_labels = raft::make_device_matrix_view<float, uint64_t>((float *)data_map["test_label"], cur_query_batch_size, l);

        auto selected_distance = raft::make_device_matrix_view<float, uint64_t>(selected_distance_device_ptr + cur_res_buff_offset * inter_buffer_size, 
                cur_query_batch_size, topk);
        auto selected_indices = raft::make_device_matrix_view<uint64_t, uint64_t>(selected_indices_device_ptr + cur_res_buff_offset * inter_buffer_size, 
                cur_query_batch_size, topk);

        cudaEventSynchronize(dis_copy_done_event[cur_res_buff_offset]);
        cudaEventSynchronize(idx_copy_done_event[cur_res_buff_offset]);

        bool run_build = query_batch_offset == 0 ? true : false;
        parafilter_build_run(dev_resources, queries, dataset,  
                          data_labels, query_labels, selected_distance, selected_indices,
                          pq_dim, n_clusters, exps, topk, global_min.data(), global_max.data(), f_config, merge_rate, run_build);

        cudaEventRecord(compute_done_event[cur_res_buff_offset]);  
        flush_current_res(selected_distance.data_handle(), selected_indices.data_handle(), inter_buffer_size, 
                      dis_copy_done_event[cur_res_buff_offset], idx_copy_done_event[cur_res_buff_offset], 
                      compute_done_event[cur_res_buff_offset], i);
        
        parafilter_mmr::free_cur_workspace_device_mems(false);
        cur_res_buff_offset = (cur_res_buff_offset + 1) % SWAP_BUFF_COUNT;
        cudaDeviceSynchronize();
      }
    }
    cudaDeviceSynchronize();
    LOG(INFO) << "device: " << i << "build time:" << build_time << ", query time:" << query_time;
    parafilter_mmr::free_cur_workspace_device_mems();
    cudaFree(selected_distance_device_ptr);
    cudaFree(selected_indices_device_ptr);
    flush_current_res((float*)0, (uint64_t*)0, 0, dis_copy_done_event[0], idx_copy_done_event[0], compute_done_event[0], 
                      i, "res/", true);
    if (data_batch_size != tot_samples) {
      uint64_t batch_size = (tot_samples + data_batch_size - 1) / data_batch_size;
      auto merged_dis_view = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, topk);
      auto merged_idx_view = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, topk);

      merge_intermediate_result<float, uint64_t>(
                                dev_resources,
                                "res/", 
                                batch_size, 
                                data_batch_size, 
                                n_queries, 
                                topk, 
                                0l, 
                                i, 
                                merged_dis_view, 
                                merged_idx_view);
      cudaEventRecord(compute_done_event[0]);
      // todo: process the case when output buffer large than the tmp buffer
      flush_current_res(merged_dis_view.data_handle(), merged_idx_view.data_handle(), 
                          topk * n_queries, dis_copy_done_event[0], idx_copy_done_event[0], compute_done_event[0], 
                          i, "res/", false, false, true);

      cudaEventSynchronize(dis_copy_done_event[0]);
      cudaEventSynchronize(idx_copy_done_event[0]);

      flush_current_res((float*)0, (uint64_t*)0, 0, dis_copy_done_event[0], idx_copy_done_event[0], compute_done_event[0], 
                      i, "res/", true, true);
      parafilter_mmr::free_cur_workspace_device_mems();
    }  
  };

  std::vector<std::thread> workers;
  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    workers.emplace_back(per_device_worker, device_id);
  }

  for(auto &w: workers) w.join();

  float recall = compute_recall("./res", std::string(dataset_path + "neighbors"), topk, tot_queries);
  LOG(INFO) << "final recall: " << recall;
  delete p_config;

  return 0;
}
