cmake_minimum_required(VERSION 3.24)

project(TrainOfThought LANGUAGES CXX)

set(TRAINOFTHOUGHT_CORE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Core"
    CACHE PATH "Path to the Core checkout")

if(NOT TARGET Core::Core)
  add_subdirectory("${TRAINOFTHOUGHT_CORE_SOURCE_DIR}"
                   "${CMAKE_CURRENT_BINARY_DIR}/Core"
                   EXCLUDE_FROM_ALL)
endif()

add_library(TrainOfThought INTERFACE)
add_library(TrainOfThought::TrainOfThought ALIAS TrainOfThought)

target_compile_features(TrainOfThought INTERFACE cxx_std_20)
target_include_directories(TrainOfThought INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(TrainOfThought INTERFACE Core::Core)

