cmake_minimum_required(VERSION 3.10)
project(CryDetection CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(ALSA REQUIRED)
find_package(FFTW3 REQUIRED)

# Update this path to point to your TFLite C library
set(TFLITE_LIBRARY "/path/to/libtensorflowlite_c_2_14_1_aarch64.so")
if(NOT EXISTS ${TFLITE_LIBRARY})
    message(FATAL_ERROR "TFLite library not found at ${TFLITE_LIBRARY}")
endif()

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    # Add any include paths for TFLite headers if needed
)

add_executable(cry_detection
    main.cpp
    CryAnnotation.h
    CryDetector.h
    CryDetector.cpp
    PredictCry.h
    PredictCry.cpp
    VADSystem.h
    VADSystem.cpp
    utils.h
    utils.cpp
)

target_link_libraries(cry_detection
    ${ALSA_LIBRARIES}
    ${FFTW3_LIBRARIES}
    ${TFLITE_LIBRARY}
    pthread
)
