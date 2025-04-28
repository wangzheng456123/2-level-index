/*
    Header for the ground truth implementation implementation for the algorrithms used in parafilter, test can be enabled if one want
    to check correctness of the algorithm.
*/
#include <cstdint>
#include <cmath>
#include <queue>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <memory>
#include "cuda_runtime.h"
#include "parafilter_utils.cuh"

template<typename ElementType>
inline ElementType min_reduce_ground_truth(ElementType* device_ptr, uint64_t size) 
{
    ElementType host_ptr = new ElementType[size];
    cudaMemcpy(host_ptr, device_ptr, size * sizeof(ElementType));
    
    ElementType res = host_ptr[0];
    for (int i = 0; i < size; i++) {
        res = std::min(res, host_ptr[i]);
    }

    return res;
}

template<typename ElementType>
inline ElementType max_reduce_ground_truth(ElementType* device_ptr, uint64_t size) 
{
    ElementType host_ptr = new ElementType[size];
    cudaMemcpy(host_ptr, device_ptr, size * sizeof(ElementType));
    
    ElementType res = host_ptr[0];
    for (int i = 0; i < size; i++) {
        res = std::max(res, host_ptr[i]);
    }

    return res;
}

// Helper function for general index calculation
template <typename IndexType>
inline IndexType calculate_index(const std::vector<IndexType>& indices, const std::vector<IndexType>& dimensions) {
    IndexType index = 0;
    IndexType multiplier = 1;

    for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
        index += indices[i] * multiplier;
        multiplier *= dimensions[i];
    }

    return index;
}

// Template function to compute LUT
template <typename ElementType, typename IndexType>
void compute_lut_cpu(
    const std::vector<ElementType>& centers,
    const std::vector<ElementType>& queries,
    std::vector<ElementType>& lut,
    IndexType pq_dim,
    IndexType n_clusters,
    IndexType pq_len,
    IndexType n_queries) 
{

    static_assert(std::is_arithmetic<ElementType>::value, "ElementType must be an arithmetic type.");
    static_assert(std::is_integral<IndexType>::value, "IndexType must be an integral type.");

    // Resize the LUT array
    lut.resize(static_cast<size_t>(n_queries) * pq_dim * n_clusters, static_cast<ElementType>(0));

    // Iterate over queries
    for (IndexType query_idx = 0; query_idx < n_queries; ++query_idx) {
        // Iterate over pq_dim
        for (IndexType pq_idx = 0; pq_idx < pq_dim; ++pq_idx) {
            // Iterate over clusters
            for (IndexType cluster_idx = 0; cluster_idx < n_clusters; ++cluster_idx) {
                // Compute the squared Euclidean distance
                ElementType distance = static_cast<ElementType>(0);
                for (IndexType i = 0; i < pq_len; ++i) {
                    /*
                    n_clusters x pq_len
                    +-----------------------------+
                    | [####] [####] [####] ...   |  <- Row 1
                    | [####] [####] [####] ...   |  <- Row 2
                    | [####] [####] [####] ...   |  <- Row 3
                    |  ...                       |
                    | [####] [####] [####] ...   |  <- Row pq_dim
                    +-----------------------------+
                    */
                    IndexType center_idx = calculate_index(
                        {pq_idx, cluster_idx, i}, 
                        {pq_dim, n_clusters, pq_len});


                    /*
                    pq_dim x pq_len
                    +-----------------------------+
                    | [####] [####] [####] ...   |  <- Row 1
                    | [####] [####] [####] ...   |  <- Row 2
                    | [####] [####] [####] ...   |  <- Row 3
                    |  ...                       |
                    | [####] [####] [####] ...   |  <- Row n_quries
                    +-----------------------------+
                    */
                    IndexType query_idx_calculated = calculate_index(
                        {query_idx, pq_idx, i}, 
                        {n_queries, pq_dim, pq_len});

                    ElementType diff = centers[center_idx] - queries[query_idx_calculated];
                    distance += diff * diff;
                }
                
                /*
                pq_dim x pq_len
                +-----------------------------+
                | [####] [####] [####] ...   |  <- Row 1
                | [####] [####] [####] ...   |  <- Row 2
                | [####] [####] [####] ...   |  <- Row 3
                |  ...                       |
                | [####] [####] [####] ...   |  <- Row n_quries
                +-----------------------------+
                */
                IndexType lut_idx = calculate_index(
                    {query_idx, pq_idx, cluster_idx}, 
                    {n_queries, pq_dim, n_clusters});

                // Store the result in LUT
                lut[lut_idx] = distance;
            }
        }
    }
}

template <typename CodebookType, typename ElementType, typename IndexType>
void compute_result_from_codebook_and_lut(
    const CodebookType* codebook,   // Codebook: n_data * pq_dim, column-major
    const ElementType* lut,        // LUT: n_queries * pq_dim * n_clusters
    ElementType* result,           // Output: n_queries * n_data
    IndexType n_data,              // Number of data points
    IndexType pq_dim,              // Number of dimensions
    IndexType n_clusters,          // Number of clusters
    IndexType n_queries)           // Number of queries
{
    static_assert(std::is_arithmetic<CodebookType>::value, "CodebookType must be arithmetic.");
    static_assert(std::is_arithmetic<ElementType>::value, "ElementType must be arithmetic.");
    static_assert(std::is_integral<IndexType>::value, "IndexType must be integral.");

    for (IndexType i = 0; i < n_data; i++) {
        for (IndexType qid = 0; qid < n_queries; qid++) {
            ElementType sum = static_cast<ElementType>(0);
            for (IndexType d = 0; d < pq_dim; d++) {
                CodebookType idx = codebook[d * n_data + i]; // Codebook in column-major order
                sum += lut[qid * pq_dim * n_clusters + d * n_clusters + idx];
            }
            result[qid * n_data + i] = sum;
        }
    }
}

template<typename ElementType, typename IndexType, typename LabelType>
void brute_forece_vector_search(ElementType* dev_dataset, 
                                ElementType* dev_queries,
                                LabelType* dev_labels,  
                                LabelType* dev_constrains, 
                                IndexType n_dim, 
                                IndexType n_queries, 
                                IndexType n_data, 
                                std::vector<std::vector<IndexType>> &neighbors,
                                int l, 
                                int topk) 
{
    ElementType* dataset = (ElementType*)malloc(n_data * n_dim * sizeof(ElementType)); 
    ElementType* queries = (ElementType*)malloc(n_queries * n_dim * sizeof(ElementType)); 
    LabelType* constrains = (LabelType*)malloc(n_queries * l * sizeof(LabelType));
    LabelType* labels = (LabelType*)malloc(n_data * l * sizeof(LabelType));

    cudaMemcpy(dataset, dev_dataset, sizeof(ElementType) * n_data * n_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(queries, dev_queries, sizeof(ElementType) * n_queries * n_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(constrains, dev_constrains, sizeof(LabelType) * n_queries * l, cudaMemcpyDeviceToHost);
    cudaMemcpy(labels, dev_labels, sizeof(LabelType) * n_data * l, cudaMemcpyDeviceToHost);
   
    for (IndexType qid = 0; qid < n_queries; qid++) {
        std::vector<LabelType> candi;
        neighbors.push_back(std::vector<IndexType>());
        for (IndexType did = 0; did < n_data; did++) {
            IndexType d_offset = did * l;
            IndexType q_offset = qid * l;
            bool is_candi = true;
            for (int l_d = 0; l_d < l; l_d++) {
                LabelType constrain = constrains[q_offset + l_d];
                LabelType le = constrain - 3 * 24 * 3600;
                LabelType ri = constrain;
                LabelType label = labels[d_offset + l_d];
                if (le > label || label > ri)
                    is_candi = false;
            }
            if (is_candi) candi.push_back(did);
        }
        std::priority_queue<std::pair<ElementType, IndexType>> pq;
        for (IndexType did : candi) {
            IndexType d_offset = did * n_dim;
            IndexType q_offset = qid * n_dim;
            ElementType cur_dis = 0;
            for (IndexType d = 0; d < n_dim; d++) {
                ElementType c_data = dataset[d_offset + d];
                ElementType q_data = queries[q_offset + d];
                cur_dis += pow(c_data - q_data, 2);
            } 
            
            pq.push(std::make_pair(cur_dis, did));
            if (pq.size() > topk) pq.pop();
        }
        for (IndexType i = 0; i < topk; i++) {
            if (pq.size()) {
                auto cur = pq.top();
                neighbors[qid].push_back(cur.second);
                pq.pop();        
            }
            else neighbors[qid].push_back(std::numeric_limits<IndexType>::max());
        }
    }

    free(dataset);
    free(queries);
    free(constrains);
    free(labels);
}

