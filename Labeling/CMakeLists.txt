add_library(lib_label src/ImagePCA.cpp include/ImagePCA.h 
                      include/ImageMethod.h src/ImageMethod.cpp 
                      include/SpatialVariance.h
                      include/CNN.h src/CNN.cpp)
target_link_libraries(lib_label ${TORCH_LIBRARIES})

add_executable(TrainingData src/TrainingData.cpp)
target_link_libraries(TrainingData lib_label lib_data)
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET TrainingData
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:TrainingData>)
endif (MSVC)

add_subdirectory(tests)