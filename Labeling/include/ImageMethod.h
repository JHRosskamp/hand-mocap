#pragma once

#include <vector>
#include "Marker.h"


class ImageMethod {
public:
  void setInputMarker(const std::vector<Marker>& in) {
    marker = in; 
  }

  void createImage() {
    project();
    normalize();
    splat();
  }

private:
  void normalize();
  void splat();
  void project();
  Eigen::Matrix3f computeProjectionMatrix();
  inCNN data;

protected:  
  virtual Eigen::Vector3f principalAxis() = 0;

  std::vector<Marker> marker, projectedMarker;
  const int IMAGE_SIZE = 52;
  const float MOCAP_CONV_INPUT_IMAGE_PADDING = 0.1;
  const float MOCAP_CONV_INPUT_DEPTH_RADIUS = std::sqrt(3.5f);
};

