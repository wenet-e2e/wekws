cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(wekws VERSION 0.1)

set(CMAKE_VERBOSE_MAKEFILE on)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g -pthread")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include(portaudio)
include(onnxruntime)
add_subdirectory(frontend)
add_subdirectory(kws)
add_subdirectory(bin)
