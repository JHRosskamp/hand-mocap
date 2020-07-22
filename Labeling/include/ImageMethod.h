#pragma once

#include <vector>
#include <string>
#include "Marker.h"


class ImageMethod {
public:
  void setInputMarker(const std::vector<Marker>& in) {
    marker = in; 
  }

  inCNN createImage() {
    project();
    normalize();
    splat();
    return data;
  }

protected:  
  void normalize();
  void splat();
  void project();
  Eigen::Matrix3f computeProjectionMatrix();
  void printImage(std::string& filename);

  virtual Eigen::Vector3f principalAxis() = 0;

  inCNN data;
  std::vector<Marker> marker, projectedMarker;
  const int IMAGE_SIZE = 52;
  const float MOCAP_CONV_INPUT_IMAGE_PADDING = 0.1;
  const float MOCAP_CONV_INPUT_DEPTH_RADIUS = std::sqrt(3.5f);
};

