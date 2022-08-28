cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(wekws VERSION 0.1)

set(CMAKE_VERBOSE_MAKEFILE on)

include(FetchContent)
include(ExternalProject)
set(FETCHCONTENT_QUIET OFF)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g -pthread")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

FetchContent_Declare(onnxruntime
  URL      https://github.com/microsoft/onnxruntime/releases/download/v1.11.1/onnxruntime-linux-x64-1.11.1.tgz
  URL_HASH SHA256=ddc03b5ae325c675ff76a6f18786ce7d310be6eb6f320087f7a0e9228115f24d
)
FetchContent_MakeAvailable(onnxruntime)
include_directories(${onnxruntime_SOURCE_DIR}/include)
link_directories(${onnxruntime_SOURCE_DIR}/lib)


add_executable(kws_main
  bin/kws_main.cc
  kws/keyword_spotting.cc
  frontend/feature_pipeline.cc
  frontend/fft.cc
)
target_link_libraries(kws_main PUBLIC onnxruntime)
