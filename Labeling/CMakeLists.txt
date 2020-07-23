add_library(lib_label src/ImagePCA.cpp include/ImagePCA.h 
                      include/ImageMethod.h src/ImageMethod.cpp 
                      include/SpatialVariance.h)

add_executable(TrainingData src/TrainingData.cpp)
target_link_libraries(TrainingData lib_label lib_data)

add_subdirectory(tests)