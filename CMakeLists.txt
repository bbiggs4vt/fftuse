cmake_minimum_required(VERSION 3.14)
project(fftuse CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# FFTW single precision (fftwf_*). The hints cover Homebrew on Apple Silicon and Intel
# Macs; elsewhere pass -DCMAKE_PREFIX_PATH=<prefix> for a non-standard install.
set(FFT_DEP_HINTS /opt/homebrew /usr/local)
find_path(FFTW_INCLUDE_DIR fftw3.h HINTS ${FFT_DEP_HINTS} PATH_SUFFIXES include REQUIRED)
find_library(FFTWF_LIBRARY NAMES fftw3f libfftw3f-3 HINTS ${FFT_DEP_HINTS} PATH_SUFFIXES lib REQUIRED)
# Header-only boost (boost::shared_ptr)
find_path(BOOST_INCLUDE_DIR boost/shared_ptr.hpp HINTS ${FFT_DEP_HINTS} PATH_SUFFIXES include REQUIRED)

# compat/ holds stand-ins for stdafx.h and dsp::utility::CFftwPlanLock
set(FFT_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/compat ${FFTW_INCLUDE_DIR} ${BOOST_INCLUDE_DIR})

function(fft_lib name)
  add_library(${name} STATIC Fft.cpp FftV2.cpp FftV3.cpp FftV4.cpp FftwWisdom.cpp)
  target_include_directories(${name} PUBLIC ${FFT_INCLUDES})
  target_link_libraries(${name} PUBLIC ${FFTWF_LIBRARY})
endfunction()

fft_lib(fftlib)

add_executable(fft_tests tests/fft_tests.cpp)
target_link_libraries(fft_tests PRIVATE fftlib)

add_executable(fft_bench bench/fft_bench.cpp)
target_link_libraries(fft_bench PRIVATE fftlib)
# Recorded in benchmark output so results from different machines are comparable
string(TOUPPER "${CMAKE_BUILD_TYPE}" FFT_BUILD_TYPE_UPPER)
target_compile_definitions(fft_bench PRIVATE
  "FFT_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\""
  "FFT_CXX_FLAGS=\"${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${FFT_BUILD_TYPE_UPPER}}\"")

add_executable(fft_wisdom tools/fft_wisdom.cpp)
target_link_libraries(fft_wisdom PRIVATE fftlib)

enable_testing()
add_test(NAME fft_tests COMMAND fft_tests)

# Bug reproduction against the original class, with sanitizers so out-of-bounds
# accesses that don't segfault still get caught. Needs fork() and GCC/Clang.
if(NOT WIN32)
  option(FFT_BUILD_REPRO "Build the sanitizer-instrumented v1 bug repro" ON)
  set(SAN -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
  if(FFT_BUILD_REPRO)
    # Some toolchains (common on embedded ARM) ship without the sanitizer runtimes
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=address,undefined")
    set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=address,undefined")
    check_cxx_source_compiles("int main() { return 0; }" FFT_HAVE_SANITIZERS)
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_LINK_OPTIONS)
    if(NOT FFT_HAVE_SANITIZERS)
      message(STATUS "Sanitizers unavailable: skipping v1_bug_repro")
      set(FFT_BUILD_REPRO OFF)
    endif()
  endif()
  if(FFT_BUILD_REPRO)
    add_library(fftlib_san STATIC Fft.cpp FftV2.cpp FftV3.cpp FftV4.cpp FftwWisdom.cpp)
    target_include_directories(fftlib_san PUBLIC ${FFT_INCLUDES})
    target_compile_options(fftlib_san PUBLIC ${SAN} -O1)
    target_link_options(fftlib_san PUBLIC ${SAN})
    target_link_libraries(fftlib_san PUBLIC ${FFTWF_LIBRARY})
    add_executable(v1_bug_repro tests/v1_bug_repro.cpp)
    target_link_libraries(v1_bug_repro PRIVATE fftlib_san)
  endif()
endif()
