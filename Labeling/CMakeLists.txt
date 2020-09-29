add_library(lib_label include/ImagePCA.h src/ImagePCA.cpp 
                      include/ImagePalm.h src/ImagePalm.cpp
                      include/ImageRandom.h src/ImageRandom.cpp
                      include/ImageMethod.h src/ImageMethod.cpp 
                      include/SpatialVariance.h
                      include/DataLoaderTotal.h 
                      include/VGG8.h
                      include/MarkerLabeling.h src/MarkerLabeling.cpp
                      include/marker_label.h src/marker_label.cpp
                      include/marker_matching.h src/marker_matching.cpp)
                      #include/BaseCNN.h 
                      #include/VGG8.h )
target_include_directories(lib_label PUBLIC ${LEMON_INCLUDE_DIRS})
target_link_libraries(lib_label ${TORCH_LIBRARIES} lemon)

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

add_executable(TrainingCNN src/CNNTraining.cpp)
target_link_libraries(TrainingCNN lib_label lib_data)

add_executable(LabelingAccuracy src/LabelingAccuracy.cpp)
target_link_libraries(LabelingAccuracy lib_label lib_data)

add_executable(Calibration src/Calibration.cpp)
target_link_libraries(Calibration lib_label lib_data lib_ik)

#add_subdirectory(tests)