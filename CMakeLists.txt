cmake_minimum_required(VERSION 3.5)

project(xllm LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2 -march=native")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

set(XLLM_CXX_SOURCES src/param.cpp src/data.cpp src/operator.cpp src/xllm.cpp src/model.cpp)

include_directories(include)
add_library(xllm OBJECT ${XLLM_CXX_SOURCES})
target_link_libraries(xllm PUBLIC ${XLLM_LINKED_LIBS})

add_executable(main main.cpp)
target_link_libraries(main xllm)

add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark xllm)

add_executable(quant src/quant.cpp)
target_link_libraries(quant xllm)

option(DEVELOPMENT OFF)
if (${DEVELOPMENT})
    message(STATUS "DEVELOPMENT MODE ON")
    add_subdirectory(tests)
endif ()
