add_library(lib_ik
	        include/IK.h src/IK.cpp "src/ik_test.cpp")
#target_link_libraries(ik_lib)

add_executable(IKTest src/ik_test.cpp)
target_link_libraries(IKTest lib_ik)