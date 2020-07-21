#include "ImagePCA.h"
#include "SpatialVariance.h"

Eigen::Vector3f ImagePCA::principalAxis() {
  
  std::vector<Eigen::Vector3f> pos;
  for (auto m : marker)
  {
    pos.push_back(m.pos);
  }
  spread<Eigen::Vector3f, Eigen::Matrix3f> pca(pos);
  pca.doEvalEvec();
  //should be normalized already
  return pca.getEigenvec().normalized();
}