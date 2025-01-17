cmake_minimum_required(VERSION 3.10)
project(CryDetection CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(ALSA REQUIRED)

# Manually specify paths for FFTW3 and TFLite
set(FFTW3_INCLUDE_DIR "$ENV{HOME}/aarch64-root/usr/include")
set(FFTW3_LIBRARY "$ENV{HOME}/aarch64-root/usr/lib/libfftw3.a")

set(TFLITE_LIBRARY "$ENV{HOME}/repos/cry_cpp_bb/cry_cpp/libtensorflowlite_c.a")
set(TFLITE_MAIN_LIB "$ENV{HOME}/repos/cry_cpp_bb/cry_cpp/libtensorflow-lite.a")

if(NOT EXISTS ${TFLITE_LIBRARY})
    message(FATAL_ERROR "TFLite library not found at ${TFLITE_LIBRARY}")
endif()

# Additional TFLite dependencies
set(XNNPACK_LIB          "$ENV{HOME}/tensorflow/build/build_static/_deps/xnnpack-build/libXNNPACK.a")
set(CPUINFO_LIB          "$ENV{HOME}/tensorflow/build/build_static/_deps/cpuinfo-build/libcpuinfo.a")
set(PTHREADPOOL_LIB      "$ENV{HOME}/tensorflow/build/build_static/pthreadpool/libpthreadpool.a")
set(FLATBUFFERS_LIB      "$ENV{HOME}/tensorflow/build/build_static/_deps/flatbuffers-build/libflatbuffers.a")
set(FARMHASH_LIB         "$ENV{HOME}/tensorflow/build/build_static/_deps/farmhash-build/libfarmhash.a")
set(FFT2D_FFTSG_LIB      "$ENV{HOME}/tensorflow/build/build_static/_deps/fft2d-build/libfft2d_fftsg.a")
set(FFT2D_FFTSG2D_LIB    "$ENV{HOME}/tensorflow/build/build_static/_deps/fft2d-build/libfft2d_fftsg2d.a")

# Ruy libraries
set(RUY_DIR "$ENV{HOME}/tensorflow/build/build_static/_deps/ruy-build/ruy")
set(RUY_LIBS
    ${RUY_DIR}/libruy_frontend.a
    ${RUY_DIR}/libruy_trmul.a
    ${RUY_DIR}/libruy_block_map.a
    ${RUY_DIR}/libruy_blocking_counter.a
    ${RUY_DIR}/libruy_thread_pool.a
    ${RUY_DIR}/libruy_apply_multiplier.a
    ${RUY_DIR}/libruy_wait.a
    ${RUY_DIR}/libruy_system_aligned_alloc.a
    ${RUY_DIR}/libruy_allocator.a
    ${RUY_DIR}/libruy_ctx.a
    ${RUY_DIR}/libruy_context.a
    ${RUY_DIR}/libruy_context_get_ctx.a
    ${RUY_DIR}/libruy_cpuinfo.a
    ${RUY_DIR}/libruy_denormal.a
    ${RUY_DIR}/libruy_tune.a
    ${RUY_DIR}/libruy_kernel_arm.a
    ${RUY_DIR}/libruy_kernel_avx.a
    ${RUY_DIR}/libruy_kernel_avx2_fma.a
    ${RUY_DIR}/libruy_kernel_avx512.a
    ${RUY_DIR}/libruy_pack_arm.a
    ${RUY_DIR}/libruy_pack_avx.a
    ${RUY_DIR}/libruy_pack_avx2_fma.a
    ${RUY_DIR}/libruy_pack_avx512.a
    ${RUY_DIR}/libruy_prepacked_cache.a
    ${RUY_DIR}/libruy_prepare_packed_matrices.a
    ${RUY_DIR}/libruy_have_built_path_for_avx.a
    ${RUY_DIR}/libruy_have_built_path_for_avx2_fma.a
    ${RUY_DIR}/libruy_have_built_path_for_avx512.a
    ${RUY_DIR}/profiler/libruy_profiler_instrumentation.a
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/lite/c
    ${FFTW3_INCLUDE_DIR}
    ${ALSA_INCLUDE_DIR}
)

#Create a stataic library
add_library(cry_detection_lib STATIC
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

# Link the library with dependencies
target_link_libraries(cry_detection_lib
    ${ALSA_LIBRARIES}
    ${FFTW3_LIBRARY}
    ${TFLITE_LIBRARY}
    ${TFLITE_MAIN_LIB}
    ${XNNPACK_LIB}
    ${CPUINFO_LIB}
    ${PTHREADPOOL_LIB}
    ${FLATBUFFERS_LIB}
    ${FARMHASH_LIB}
    ${FFT2D_FFTSG_LIB}
    ${FFT2D_FFTSG2D_LIB}
    -Wl,--start-group
    ${RUY_LIBS}
    -Wl,--end-group
    pthread
    dl
    m
)

# Create executable
add_executable(cry_detection
    main.cpp
)

# Link the executable with dependencies
target_link_libraries(cry_detection
    cry_detection_lib
)
