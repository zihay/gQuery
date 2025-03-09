# Copyright 2024 gQuery Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.15...3.27)
project(gquery)

enable_testing()

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# import the nanobind package
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# import nanobind https://nanobind.readthedocs.io/en/latest/building.html
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# import eigen
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0  # or any specific tag/commit you need
)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.12.1.zip
)

# Prevent overriding parent's compiler/linker settings on Windows.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Find or fetch spdlog with bundled fmt
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.12.0  # Use a stable version
    )
    # Disable spdlog tests and examples
    set(SPDLOG_BUILD_EXAMPLE OFF CACHE INTERNAL "")
    set(SPDLOG_BUILD_TESTS OFF CACHE INTERNAL "")
    set(SPDLOG_INSTALL OFF CACHE INTERNAL "")
    # Use spdlog's bundled fmt (this is the default, so we don't need to set it)
    # set(SPDLOG_FMT_EXTERNAL OFF CACHE INTERNAL "")
    FetchContent_MakeAvailable(spdlog)
endif()

include_directories(include)

add_library(gquery STATIC
  src/util/log.cpp
  src/util/check.cpp    
  src/util/print.cpp
  src/core/bvh.cpp
)
target_link_libraries(gquery PUBLIC 
    Eigen3::Eigen 
    spdlog::spdlog
)

nanobind_add_module(
  # Name of the extension
  gquery_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC
  src/gquery.cpp
)
target_link_libraries(gquery_ext PRIVATE gquery)

if(APPLE)
  set_target_properties(gquery_ext PROPERTIES
      INSTALL_RPATH "@loader_path/../lib")
endif()

nanobind_add_stub(
    gquery_ext_stub
    MODULE gquery_ext
    OUTPUT gquery_ext.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:gquery_ext>
    MARKER_FILE py.typed
    DEPENDS gquery_ext
)

install(TARGETS gquery_ext LIBRARY DESTINATION gquery)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/py.typed 
              ${CMAKE_CURRENT_BINARY_DIR}/gquery_ext.pyi DESTINATION gquery)

# ------------------------------------------------------------------------------
# Add tests
# ------------------------------------------------------------------------------
include(GoogleTest)

# Define all test files
set(TEST_SOURCES
  src/tests/test_main.cpp
  # Add additional test files here, e.g.:
  # src/tests/test_core.cpp
  # src/tests/test_util.cpp
)

# Create test executable
add_executable(gquery_test ${TEST_SOURCES})

# Link dependencies
target_link_libraries(gquery_test 
  PRIVATE 
    gquery 
    gtest_main
    spdlog::spdlog
)

# Enable CTest integration with Google Test discovery
gtest_discover_tests(gquery_test)

# Optional: Add a custom target for running tests with more detailed output
add_custom_target(test_verbose
  COMMAND ${CMAKE_CTEST_COMMAND} --verbose
  DEPENDS gquery_test
)


