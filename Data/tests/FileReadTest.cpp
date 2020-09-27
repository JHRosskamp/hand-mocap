#include "MarkerFileInput.h"
#include <string>
#include <Eigen/Dense>
#include <catch2/catch.hpp>

TEST_CASE("Read Training Data") {
  std::string filePath = "..\\TestData\\training_data.trc";
  MarkerFileInput input(filePath);

  SECTION("split test") {
    std::string s = "a\tbc\tdef";
    auto res = splitString(s);
    REQUIRE(res[0] == "a");
    REQUIRE(res[1] == "bc");
    REQUIRE(res[2] == "def");
  }

  input.readFile();
  metadata meta = input.getMetadata();
  REQUIRE(meta.data_rate == 120);
  REQUIRE(meta.units == "mm");

  auto numberFrames = input.numberOfFrames();
  REQUIRE(numberFrames == 51);

  std::vector<Marker> marker = input.getNextFrame();
  REQUIRE(marker.size() == 19);
  REQUIRE(marker[0].pos.x() == Approx(-137.943));
  REQUIRE(marker[18].pos.z() == Approx(90.2687));

  for (int i = 1; i < numberFrames; ++i) {
    std::vector<Marker> marker = input.getNextFrame();
    if (i == numberFrames - 1) {
      REQUIRE(marker[0].pos.y() == Approx(79.3045));
      REQUIRE(marker[18].pos.z() == Approx(95.0649));
    }
  }

}
