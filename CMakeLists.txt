#cmake_minimum_required (VERSION 3.8)
project ("HandTracking")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(UNREAL_PLUGIN_PATH "D:/UETracking/Plugins/OptiTracking")
set(UNREAL_LIB_DIR "${UNREAL_PLUGIN_PATH}/ThirdParty/Tracking")
set(UNREAL_INCLUDE_DIR "${UNREAL_PLUGIN_PATH}/Source/OptiTracking/Public")

enable_testing()

# define output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(PYTORCH_PATH "D:/Libs/libtorch/")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PYTORCH_PATH})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(HAND_TRACKING_LEMON_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/cmake/lemon.cmake")
if (EXISTS ${HAND_TRACKING_LEMON_CONFIG})
    include(${HAND_TRACKING_LEMON_CONFIG})
else ()
    find_package(LEMON REQUIRED)
endif ()
add_library(lemon STATIC IMPORTED)
set_target_properties(lemon PROPERTIES
		IMPORTED_LOCATION "${LEMON_LIBRARIES}")
target_include_directories(lemon INTERFACE ${LEMON_INCLUDE_DIRS})


include_directories(${CMAKE_SOURCE_DIR}/ThirdParty/includes)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/Labeling/include)
include_directories(${CMAKE_SOURCE_DIR}/Data/include)
include_directories(${CMAKE_SOURCE_DIR}/IK/include)

add_subdirectory(Labeling)
add_subdirectory(Data)
add_subdirectory(interfaces)
add_subdirectory(IK)
add_subdirectory(main)