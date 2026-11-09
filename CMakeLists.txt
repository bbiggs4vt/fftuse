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

# std::mutex needs libpthread on older glibc (common on embedded ARM distros).
# libm covers a static libfftw3f.a, e.g. a self-built NEON FFTW.
find_package(Threads REQUIRED)
set(FFT_LINK_LIBS ${FFTWF_LIBRARY} Threads::Threads)
if(UNIX AND NOT APPLE)
  list(APPEND FFT_LINK_LIBS m)
endif()

# Flags for CFftV4's kernels only (FftV4Kernels.cpp); everything else keeps the
# toolchain's flags. See FftV4Kernels.h.
#
# Always, where supported: -ftree-vectorize -fvect-cost-model=dynamic. GCC's -O2
# (the default in Yocto and many distro builds) uses the "very cheap" cost model,
# which leaves all four kernel loops scalar; -O3 or these flags vectorize them.
#
# FFT_ARMV7_NEON (32-bit ARM, e.g. Cortex-A8/A9): adds -mfpu=neon
# -funsafe-math-optimizations. ARMv7 GCC only vectorizes float loops for NEON with
# the latter. The resulting binary needs a CPU with NEON.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-ftree-vectorize -fvect-cost-model=dynamic" FFT_HAVE_VECT_COST_MODEL)
# With link-time optimization (-flto, e.g. CMAKE_INTERPROCEDURAL_OPTIMIZATION),
# the kernels are inlined into callers built without the flags below and end up
# scalar. -fno-lto keeps them as separate, vectorized functions.
check_cxx_compiler_flag("-fno-lto" FFT_HAVE_NO_LTO)
option(FFT_ARMV7_NEON "Build CFftV4's kernels with -mfpu=neon -funsafe-math-optimizations (32-bit ARM)" OFF)
set(FFT_V4_KERNEL_FLAGS "")
if(FFT_HAVE_VECT_COST_MODEL)
  list(APPEND FFT_V4_KERNEL_FLAGS -ftree-vectorize -fvect-cost-model=dynamic)
endif()
if(FFT_ARMV7_NEON)
  list(APPEND FFT_V4_KERNEL_FLAGS -mfpu=neon -funsafe-math-optimizations)
endif()
if(FFT_HAVE_NO_LTO)
  list(APPEND FFT_V4_KERNEL_FLAGS -fno-lto)
endif()
if(FFT_V4_KERNEL_FLAGS)
  set_source_files_properties(FftV4Kernels.cpp PROPERTIES COMPILE_OPTIONS "${FFT_V4_KERNEL_FLAGS}")
endif()
string(REPLACE ";" " " FFT_V4_KERNEL_FLAGS_TEXT "${FFT_V4_KERNEL_FLAGS}")

set(FFT_SOURCES Fft.cpp FftV2.cpp FftV3.cpp FftV4.cpp FftV4Kernels.cpp FftwWisdom.cpp)

function(fft_lib name)
  add_library(${name} STATIC ${FFT_SOURCES})
  target_include_directories(${name} PUBLIC ${FFT_INCLUDES})
  target_link_libraries(${name} PUBLIC ${FFT_LINK_LIBS})
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
  "FFT_CXX_FLAGS=\"${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${FFT_BUILD_TYPE_UPPER}}\""
  "FFT_V4_KERNEL_FLAGS=\"${FFT_V4_KERNEL_FLAGS_TEXT}\"")

add_executable(fft_wisdom tools/fft_wisdom.cpp)
target_link_libraries(fft_wisdom PRIVATE fftlib)

add_executable(fft_plan_check tools/fft_plan_check.cpp)
target_link_libraries(fft_plan_check PRIVATE fftlib)

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
    add_library(fftlib_san STATIC ${FFT_SOURCES})
    target_include_directories(fftlib_san PUBLIC ${FFT_INCLUDES})
    target_compile_options(fftlib_san PUBLIC ${SAN} -O1)
    target_link_options(fftlib_san PUBLIC ${SAN})
    target_link_libraries(fftlib_san PUBLIC ${FFT_LINK_LIBS})
    add_executable(v1_bug_repro tests/v1_bug_repro.cpp)
    target_link_libraries(v1_bug_repro PRIVATE fftlib_san)
  endif()
endif()
