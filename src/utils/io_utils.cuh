#pragma once
#include <cstdint>
#include <fstream>
#include <future> 
#include <filesystem>
#include <vector>
#include <map>
#include <unordered_map>
#include <iostream>
#include <assert.h>
#include <stdlib.h>
#include <core/mmr.cuh>
#include "easylogging++.h"

void get_data_set_list(std::vector<std::string> &keys, const char file[]);
void get_data_type_list(std::map<std::string, std::string> &data_type, const char file[], 
    std::vector<std::string> const& keys);
void* read_binary_file(const std::string& file_path, uint64_t offset, uint64_t size);
std::future<bool> write_binary_file_async(const std::string& file_path, uint64_t offset, const void* data, uint64_t size, bool append = true);
void load_res_files_from_directory(
    const std::string& directory_path, 
    std::vector<float>& distances,
    std::vector<uint64_t>& neighbors);
void build_dataset(
        std::vector<std::string> const& keys,
        std::map<std::string, std::string> const& data_type,
        std::map<std::string, void*>* data_map,
        std::map<std::string, std::pair<int32_t, int32_t>>& size_map,
        const std::string &dir,
        uint64_t data_offset = 0,
        uint64_t data_batch_size = 0,
        uint64_t query_offset = 0,
        uint64_t query_batch_size = 0, 
        int filter_dim = 1);


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