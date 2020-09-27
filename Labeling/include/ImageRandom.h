#pragma once
#include "ImageMethod.h"

class ImageRandom : public ImageMethod {
public:
  ImageRandom() {};
  Eigen::Vector3f principalAxis();

};