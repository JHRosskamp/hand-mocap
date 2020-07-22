#include <catch2/catch.hpp>
#include <iostream>
#include "ImageMethod.h"
#include "Marker.h"

class imageImpl : public ImageMethod {
public:
  imageImpl() {
    inMarker.push_back(Marker(Eigen::Vector3f(0, 0, 0), marker_label::thumb_tip));
    inMarker.push_back(Marker(Eigen::Vector3f(1, 0, 0), marker_label::index_tip));
    inMarker.push_back(Marker(Eigen::Vector3f(1, 1, 0), marker_label::middle_tip));
    inMarker.push_back(Marker(Eigen::Vector3f(0, 1, 0), marker_label::ring_tip));
    inMarker.push_back(Marker(Eigen::Vector3f(0.5, 0.5, 1), marker_label::little_tip));

    setInputMarker(inMarker);
  }
  Eigen::Vector3f principalAxis() {
    return Eigen::Vector3f(0.0, 0.0, 1.0);
  }

  void callProject() {
    project();
  }

  void callNormalize() {
    normalize();
  }

  void callSplat() {
    splat();
  }

  void callPrint() {
    std::string filename = "test_depth_image.pgm";
    printImage(filename);
  }

  std::vector<Marker> getProjectedMarker() {
    return projectedMarker;
  }

  inCNN getOutData() {
    return data;
  }

  std::vector<Marker> inMarker;
};

TEST_CASE("Standard Image Generation Test") {
  imageImpl testObject = imageImpl();
  
  testObject.callProject();
  SECTION("Check projections") {
    std::vector<Marker> projectedMarker = testObject.getProjectedMarker();
    REQUIRE(projectedMarker[0].pos.x() == 0.0);
    //Depth information are on y-axis
    REQUIRE(projectedMarker[3].pos.x() == 0.0);
    REQUIRE(projectedMarker[3].pos.y() == 0.0);
    REQUIRE(projectedMarker[3].pos.z() == 1.0);
    REQUIRE(projectedMarker[4].pos.y() == 1.0);
  }
  testObject.callNormalize();
  SECTION("Check normalization") {
    inCNN data = testObject.getOutData();
    //Depth
    REQUIRE(data.normalized_marker[0].pos.y() == Approx(0.1f));
    REQUIRE(data.normalized_marker[4].pos.y() == Approx(1.f));
    //spatial
    REQUIRE(data.normalized_marker[0].pos.x() == Approx(0.1f));
    REQUIRE(data.normalized_marker[1].pos.x() == Approx(0.9f));
  }
  testObject.callSplat();
  SECTION("Check labels") {
    inCNN data = testObject.getOutData();
    REQUIRE(data.normalized_marker[0].label == marker_label::thumb_tip);
    REQUIRE(data.normalized_marker[4].label == marker_label::little_tip);
  }

  testObject.callPrint();
}