# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# ------------- configure rapids-cmake --------------#

include(cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#

rapids_cuda_init_architectures(test_raft)

project(test_raft LANGUAGES CXX CUDA)

# ------------- configure raft -----------------#

rapids_cpm_init()
include(cmake/thirdparty/get_raft.cmake)

# -------------- compile tasks ----------------- #
file(GLOB_RECURSE INDEX "src/index/*.cu")
file(GLOB_RECURSE CORE "src/core/*.cu")
file(GLOB_RECURSE SEARCH-ENGINE "src/search-engine/*.cu")
file(GLOB_RECURSE UTILS "src/utils/*.cu")
file(GLOB_RECURSE INDEX_SECONDARY "src/index/2-level_index/*.cu")

add_library(easylogging src/easylogging++.cc)
target_compile_definitions(easylogging PUBLIC ELPP_THREAD_SAFE)

add_executable(search-engine src/auto-batch-search-main.cu ${INDEX} ${CORE} ${SEARCH-ENGINE} ${UTILS} ${INDEX_SECONDARY})
target_include_directories(search-engine PRIVATE src)
target_link_libraries(search-engine PRIVATE raft::raft raft::compiled easylogging)

