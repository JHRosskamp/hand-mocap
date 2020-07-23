add_executable(FileReadTest MainTest.cpp FileReadTest.cpp)
target_link_libraries(FileReadTest lib_data)

#include(CTest)
#include(Catch)
#include(ParseAndAddCatchTests)
#catch_discover_tests(ImageTest)