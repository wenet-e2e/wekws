cmake_minimum_required(VERSION 3.4.1)
project(wekws CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE on)

set(build_DIR ${CMAKE_SOURCE_DIR}/../../../build)
file(GLOB ONNXRUNTIME_INCLUDE_DIRS "${build_DIR}/onnxruntime*.aar/headers")
file(GLOB ONNXRUNTIME_LINK_DIRS "${build_DIR}/onnxruntime*.aar/jni/${ANDROID_ABI}")
link_directories(${ONNXRUNTIME_LINK_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR})
add_library(wekws SHARED
  frontend/feature_pipeline.cc
  frontend/fft.cc
  kws/keyword_spotting.cc
  wekws.cc
)
target_link_libraries(wekws PUBLIC onnxruntime)
