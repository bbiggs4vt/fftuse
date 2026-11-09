cmake_minimum_required(VERSION 3.14)
project(fftuse CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# FFTW single precision (fftwf_*)
find_path(FFTW_INCLUDE_DIR fftw3.h REQUIRED)
find_library(FFTWF_LIBRARY NAMES fftw3f libfftw3f-3 REQUIRED)
# Header-only boost (boost::shared_ptr)
find_path(BOOST_INCLUDE_DIR boost/shared_ptr.hpp REQUIRED)

# compat/ holds stand-ins for stdafx.h and dsp::utility::CFftwPlanLock
set(FFT_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/compat ${FFTW_INCLUDE_DIR} ${BOOST_INCLUDE_DIR})

function(fft_lib name)
  add_library(${name} STATIC Fft.cpp FftV2.cpp FftV3.cpp)
  target_include_directories(${name} PUBLIC ${FFT_INCLUDES})
  target_link_libraries(${name} PUBLIC ${FFTWF_LIBRARY})
endfunction()

fft_lib(fftlib)

add_executable(fft_tests tests/fft_tests.cpp)
target_link_libraries(fft_tests PRIVATE fftlib)

add_executable(fft_bench bench/fft_bench.cpp)
target_link_libraries(fft_bench PRIVATE fftlib)

enable_testing()
add_test(NAME fft_tests COMMAND fft_tests)

# Bug reproduction against the original class, with sanitizers so out-of-bounds
# accesses that don't segfault still get caught. Needs fork() and GCC/Clang.
if(NOT WIN32)
  option(FFT_BUILD_REPRO "Build the sanitizer-instrumented v1 bug repro" ON)
  if(FFT_BUILD_REPRO)
    set(SAN -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
    add_library(fftlib_san STATIC Fft.cpp FftV2.cpp FftV3.cpp)
    target_include_directories(fftlib_san PUBLIC ${FFT_INCLUDES})
    target_compile_options(fftlib_san PUBLIC ${SAN} -O1)
    target_link_options(fftlib_san PUBLIC ${SAN})
    target_link_libraries(fftlib_san PUBLIC ${FFTWF_LIBRARY})
    add_executable(v1_bug_repro tests/v1_bug_repro.cpp)
    target_link_libraries(v1_bug_repro PRIVATE fftlib_san)
  endif()
endif()
