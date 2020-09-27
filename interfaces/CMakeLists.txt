add_library(TrackingLib SHARED iue_interface.h
    ue_interface.h ue_interface.cpp)
target_link_libraries(TrackingLib ${TORCH_LIBRARIES} lib_label lib_data lib_ik)
#install(TARGETS TrackingLib DESTINATION lib)

add_custom_command(TARGET TrackingLib POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo 
  "target dir = $<TARGET_FILE_DIR:TrackingLib>")

add_custom_command(TARGET TrackingLib POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${PROJECT_SOURCE_DIR}/bin/TrackingLib.dll"
  "${PROJECT_SOURCE_DIR}/bin/TrackingLib.lib"
  "${UNREAL_LIB_DIR}")

add_custom_command(TARGET TrackingLib POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${PROJECT_SOURCE_DIR}/interfaces/iue_interface.h"
  "${UNREAL_INCLUDE_DIR}")