#pragma once
#include "ImageMethod.h"

class ImagePalm : public ImageMethod {
public:
  ImagePalm() {};
  Eigen::Vector3f principalAxis();

};