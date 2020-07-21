add_library(lib_label src/ImagePCA.cpp include/ImagePCA.h include/ImageMethod.h "src/ImageMethod.cpp" "include/SpatialVariance.h")


add_subdirectory(tests)