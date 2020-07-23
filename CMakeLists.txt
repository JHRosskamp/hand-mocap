#cmake_minimum_required (VERSION 3.8)
project ("HandTracking")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

# define output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(PYTORCH_PATH "D:/Libs/libtorch/")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PYTORCH_PATH})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${CMAKE_SOURCE_DIR}/ThirdParty/includes)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/Labeling/include)
include_directories(${CMAKE_SOURCE_DIR}/Data/include)

add_subdirectory(Labeling)
add_subdirectory(Data)