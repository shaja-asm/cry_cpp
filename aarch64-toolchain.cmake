# Cross-compilation toolchain for aarch64 using GCC 9
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Specify GCC 9 cross compiler
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc-9)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++-9)

# Set the target sysroot (recommended for robust builds)
set(CMAKE_SYSROOT "$ENV{HOME}/aarch64-root")

# ALSA paths
set(ALSA_INCLUDE_DIR "$ENV{HOME}/aarch64-root/usr/include")
set(ALSA_LIBRARY "$ENV{HOME}/aarch64-root/usr/lib/libasound.a")

# FFTW3 paths
set(FFTW3_INCLUDE_DIR "$ENV{HOME}/aarch64-root/usr/include")
set(FFTW3_LIBRARY "$ENV{HOME}/aarch64-root/usr/lib/libfftw3.a")

# JNI paths
set(JNI_INCLUDE_DIR "$ENV{HOME}/aarch64-root/usr/include/jni")
include_directories(${JNI_INCLUDE_DIR})

# Print compiler details for debugging
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
                OUTPUT_VARIABLE CMAKE_CXX_COMPILER_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "C++ Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")

# Static linking flags (optional)
set(CMAKE_EXE_LINKER_FLAGS "-static")

# Search only within the sysroot for libraries and includes
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
