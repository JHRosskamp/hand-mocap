#include "ImageMethod.h"

class ImagePCA : public ImageMethod {
public:
  ImagePCA() {};
  Eigen::Vector3f principalAxis();

};