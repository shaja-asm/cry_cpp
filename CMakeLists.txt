cmake_minimum_required(VERSION 3.10)
project(CryDetection CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(ALSA REQUIRED)

# Manually specify paths for FFTW3
set(FFTW3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../aarch64-root/usr/include")
set(FFTW3_LIBRARY "${CMAKE_SOURCE_DIR}/../aarch64-root/usr/lib/libfftw3.a")

# Update this path to point to your TFLite C library
set(TFLITE_LIBRARY "/home/meow/repos/cry_cpp/libtensorflowlite_c_2_14_1_aarch64.so")
if(NOT EXISTS ${TFLITE_LIBRARY})
    message(FATAL_ERROR "TFLite library not found at ${TFLITE_LIBRARY}")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/lite/c
    ${FFTW3_INCLUDE_DIR}
)

# Source files
add_executable(cry_detection
    main.cpp
    CryDetector.cpp
    PredictCry.cpp
    VADSystem.cpp
    utils.cpp
    tensorflow/lite/c/c_api_for_testing.cc
    tensorflow/lite/c/c_api_opaque_internal.cc
    tensorflow/lite/c/common_internal.cc
    tensorflow/lite/c/test_util.cc
    tensorflow/lite/c/jni/jni_utils.cc
)

# Link libraries
target_link_libraries(cry_detection
    ${ALSA_LIBRARIES}
    ${FFTW3_LIBRARY}
    ${TFLITE_LIBRARY}
    pthread
)
