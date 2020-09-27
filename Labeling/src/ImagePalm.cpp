#include "ImagePalm.h"
#include "SpatialVariance.h"

Eigen::Vector3f ImagePalm::principalAxis() {
  
  std::vector<Eigen::Vector3f> pos;
  for (auto m : marker)
  {
    pos.push_back(m.pos);
  }
  //spread<Eigen::Vector3f, Eigen::Matrix3f> pca(pos);
  //pca.doEvalEvec();
  //should be normalized already
  Eigen::Vector3f vec1 = pos[17] - pos[16];
  Eigen::Vector3f vec2 = pos[18] - pos[16];
  return vec1.cross(vec2).normalized();
}