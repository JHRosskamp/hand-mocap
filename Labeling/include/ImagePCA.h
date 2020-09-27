#pragma once
#include "ImageMethod.h"

class ImagePCA : public ImageMethod {
public:
  ImagePCA() {};
  Eigen::Vector3f principalAxis();
  std::vector<Eigen::Vector3f> getAllAxis();

};