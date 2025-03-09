cmake_minimum_required(VERSION 3.15...3.27)

# ============================================================================
# Project configuration
# ============================================================================
project(gquery 
  VERSION 0.1.0
  DESCRIPTION "A geometric query library"
  LANGUAGES CXX
)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS 
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# C++ standard requirements
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enable testing
enable_testing()

# ============================================================================
# Dependencies
# ============================================================================
include(FetchContent)

# --- Python ---
if(CMAKE_VERSION VERSION_LESS 3.18)
  set(PYTHON_DEV_MODULE Development)
else()
  set(PYTHON_DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${PYTHON_DEV_MODULE} REQUIRED)

# --- nanobind ---
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE 
  OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

# --- Eigen ---
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# --- Google Test ---
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.12.1.zip
)
# Prevent overriding parent's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --- spdlog ---
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
  # Disable spdlog tests and examples
  set(SPDLOG_BUILD_EXAMPLE OFF CACHE INTERNAL "")
  set(SPDLOG_BUILD_TESTS OFF CACHE INTERNAL "")
  set(SPDLOG_INSTALL OFF CACHE INTERNAL "")
  
  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
  )
  FetchContent_MakeAvailable(spdlog)
endif()

# ============================================================================
# Library
# ============================================================================
include_directories(include)

# Core library
add_library(gquery STATIC
  src/util/log.cpp
  src/util/check.cpp    
  src/util/print.cpp
  src/core/bvh.cpp
  src/core/parallel.cpp
)

target_link_libraries(gquery PUBLIC 
  Eigen3::Eigen 
  spdlog::spdlog
)

# ============================================================================
# Python Extension
# ============================================================================
nanobind_add_module(
  gquery_ext
  STABLE_ABI     # Target the stable ABI for Python 3.12+
  NB_STATIC      # Build libnanobind statically and merge it into the extension
  src/gquery.cpp
)

target_link_libraries(gquery_ext PRIVATE gquery)

# Platform-specific settings
if(APPLE)
  set_target_properties(gquery_ext PROPERTIES
    INSTALL_RPATH "@loader_path/../lib"
  )
endif()

# Generate Python stub file for better IDE integration
nanobind_add_stub(
  gquery_ext_stub
  MODULE gquery_ext
  OUTPUT gquery_ext.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:gquery_ext>
  MARKER_FILE py.typed
  DEPENDS gquery_ext
)

# ============================================================================
# Tests
# ============================================================================
include(GoogleTest)

# Test executable
add_executable(gquery_test
  src/tests/test_main.cpp
  src/tests/test_parallel.cpp
)

target_link_libraries(gquery_test PRIVATE 
  gquery 
  gtest_main
  spdlog::spdlog
)

# Enable CTest integration with Google Test discovery
gtest_discover_tests(gquery_test)

# Add a custom target for running tests with more detailed output
add_custom_target(test_verbose
  COMMAND ${CMAKE_CTEST_COMMAND} --verbose
  DEPENDS gquery_test
)

# ============================================================================
# Installation
# ============================================================================
install(TARGETS gquery_ext 
  LIBRARY DESTINATION gquery
)

install(FILES 
  ${CMAKE_CURRENT_BINARY_DIR}/py.typed 
  ${CMAKE_CURRENT_BINARY_DIR}/gquery_ext.pyi 
  DESTINATION gquery
)


