cmake_minimum_required (VERSION 3.8)
project ("HandTracking")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# define output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

#set(EXT_INCLUDES "${CMAKE_SOURCE_DIR}/ThirdParty/includes")
include_directories(${CMAKE_SOURCE_DIR}/ThirdParty/includes)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/Labeling/include)

add_subdirectory(Labeling)