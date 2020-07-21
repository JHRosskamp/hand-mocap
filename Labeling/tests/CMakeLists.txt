add_executable(ImageTest MainTest.cpp ImageTest.cpp)
target_link_libraries(ImageTest lib_label)

#include(CTest)
#include(Catch)
#include(ParseAndAddCatchTests)
#catch_discover_tests(ImageTest)