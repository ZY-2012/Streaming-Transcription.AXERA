cmake_minimum_required(VERSION 3.15)
project(nemotron_ax LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# BSP layout: <BSP_MSP_DIR>/include/ax_engine_api.h and <BSP_MSP_DIR>/lib/*.so
if(NOT BSP_MSP_DIR)
  set(BSP_MSP_DIR "$ENV{BSP_MSP_DIR}")
endif()
if(NOT BSP_MSP_DIR)
  message(FATAL_ERROR
    "Set BSP_MSP_DIR to <ax650n_bsp_sdk>/msp/out "
    "(see cpp/README.md / download_toolchains.sh)")
endif()

include_directories(
  ${BSP_MSP_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kissfft
)
link_directories(${BSP_MSP_DIR}/lib)

find_package(Threads REQUIRED)

# kissfft is C, so it needs its own target (a C++-only project silently skips it).
add_library(nemotron_kissfft STATIC third_party/kissfft/kiss_fft.c)
target_include_directories(nemotron_kissfft PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kissfft)

add_library(nemotron_sdk STATIC
  src/audio_reader.cpp
  src/npy_reader.cpp
  src/mel_frontend.cpp
  src/ax_engine_runner.cpp
  src/nemotron_state.cpp
  src/nemotron_postprocess.cpp
  src/nemotron_diarizer.cpp
)
# AX_NPU_* symbols live in libax_interpreter, not libax_engine.
target_link_libraries(nemotron_sdk PUBLIC
  nemotron_kissfft ax_engine ax_interpreter ax_sys Threads::Threads dl m)

add_executable(nemotron_ax650 nemotron_ax.cpp)
target_link_libraries(nemotron_ax650 PRIVATE nemotron_sdk)

add_executable(bench_encoder tools/bench_encoder.cpp)
target_link_libraries(bench_encoder PRIVATE nemotron_sdk)
