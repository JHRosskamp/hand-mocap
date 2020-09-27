#include "ImageRandom.h"
#include "SpatialVariance.h"
#include <random>

Eigen::Vector3f ImageRandom::principalAxis() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  Eigen::Vector3f camera_pos = Eigen::Vector3f(dis(gen), dis(gen), dis(gen));

  return camera_pos.normalized();
}