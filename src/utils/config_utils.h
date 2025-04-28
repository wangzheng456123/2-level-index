#pragma once
#include "easylogging++.h"
#include <fstream>
#include <utils/string_utils.h>
#include <stdint.h>
#include <functional>

#define OFFSET_OF(TYPE, MEMBER) \
    (reinterpret_cast<std::size_t>(&reinterpret_cast<const volatile char&>(static_cast<TYPE*>(nullptr)->MEMBER)))
#define OFFSETOF_NESTED(T1, M1, T2, M2) (OFFSET_OF(T1, M1) + OFFSET_OF(T2, M2))

struct common_config {
    uint64_t data_width;
    uint64_t index_width;
    // parameters for multiple rounds filtering
    uint64_t topk;
    uint64_t break_down;
    uint64_t enable_multi_gpu;
    uint64_t mem_bound;
    uint64_t is_calc_mem_predictor_coeff;
    uint64_t lowest_query_batch;

    std::string path;

    common_config(std::string const& path_to_config, std::function<void(std::map<std::string, int>&)> extender) :
        path_to_config(path_to_config) 
    {
        init_str_to_offset_map_common();
        extender(str_to_offset_map);
    }

    void read_config() {
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
                    *(static_cast<uint64_t *>(static_cast<void*>((reinterpret_cast<char*>(static_cast<void*>(this))) + 
                            str_to_offset_map[currConfigStr])))
                        = (uint64_t)(std::atoll(currValue.c_str()));
                }
                else {
                    *(static_cast<float *>(static_cast<void*>((reinterpret_cast<char*>(static_cast<void*>(this))) + 
                            str_to_offset_map[currConfigStr])))
                        = (float)(std::stof(currValue.c_str()));
                }
            }
            else {
                *(static_cast<std::string *>(static_cast<void*>((reinterpret_cast<char*>(static_cast<void*>(this))) +  
                            str_to_offset_map[currConfigStr])))
                    = currValue;
            }
        }
    }

protected:
    /*initialize string to offset data in class static function*/
    std::map<std::string, int> str_to_offset_map;
    std::string path_to_config;
private:
    void init_str_to_offset_map_common()
    {
        str_to_offset_map = { 
            {"DATA_WIDTH", OFFSET_OF(common_config, data_width)}, 
            {"INDEX_WIDTH", OFFSET_OF(common_config, index_width)}, 
            {"PATH", OFFSET_OF(common_config, path)}, 
            {"TOPK", OFFSET_OF(common_config, topk)}, 
            {"BREAK_DOWN", OFFSET_OF(common_config, break_down)}, 
            {"ENABLE_MULTI_GPU", OFFSET_OF(common_config, enable_multi_gpu)}, 
            {"MEM_BOUND", OFFSET_OF(common_config, mem_bound)}, 
            {"IS_CALC_MEM_PREDICTOR_COEFF", OFFSET_OF(common_config, is_calc_mem_predictor_coeff)}, 
            {"LOWEST_QUERY_BATCH", OFFSET_OF(common_config, lowest_query_batch)}, 
      };
    }
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
