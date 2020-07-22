#include "ImageMethod.h"
#include <iostream>
#include <fstream>

void ImageMethod::normalize() {
  const size_t nMarkers = projectedMarker.size();
  data.normalized_marker.resize(nMarkers);

  float xMin = FLT_MAX;
  float xMax = -FLT_MAX;
  float zMin = FLT_MAX;
  float zMax = -FLT_MAX;
  float depthMin = FLT_MAX;
  float depthMax = -FLT_MIN;

  //write better some day
  for (size_t iMarker = 0; iMarker < nMarkers; ++iMarker) {
    xMin = std::min(projectedMarker[iMarker].pos.x(), xMin);
    zMin = std::min(projectedMarker[iMarker].pos.z(), zMin);
    xMax = std::max(projectedMarker[iMarker].pos.x(), xMax);
    zMax = std::max(projectedMarker[iMarker].pos.z(), zMax);
    depthMin = std::min(projectedMarker[iMarker].pos.y(), depthMin);
    depthMax = std::max(projectedMarker[iMarker].pos.y(), depthMax);
  }

  const float shift_x = -xMin;
  const float shift_z = -zMin;
  const float shift_depth = -depthMin;
  const float scale_axis = std::max(xMax + shift_x, zMax + shift_z) / 0.8f;
  const float scale_depth = (depthMax + shift_depth) / 0.9f; //skaliere zwischen 0 und 0.9 um anschließend um +0.1 zu verschieben

  for (size_t iMarker = 0; iMarker < nMarkers; ++iMarker)
  {
    Eigen::Vector2f plane = Eigen::Vector2f(projectedMarker[iMarker].pos.x() + shift_x, projectedMarker[iMarker].pos.z() + shift_z) / scale_axis + Eigen::Vector2f(0.1f, 0.1f);
    float depth = (projectedMarker[iMarker].pos.y() + shift_depth) / scale_depth + 0.1f;
    data.normalized_marker[iMarker] = Marker(Eigen::Vector3f(plane.x(), depth, plane.y()), projectedMarker[iMarker].label);
  }
}

void ImageMethod::splat() {
  const size_t nMarkers = marker.size();
  std::vector<float> depthMap(IMAGE_SIZE * IMAGE_SIZE, 0.0);

  const long searchRadiusInPixel = (long)ceil(MOCAP_CONV_INPUT_DEPTH_RADIUS + 1);
  const float dist2Threshold = std::pow(MOCAP_CONV_INPUT_DEPTH_RADIUS, 2);

  for (size_t iMarker = 0; iMarker < nMarkers; ++iMarker) {
    Eigen::Vector2f imgPos = Eigen::Vector2f(data.normalized_marker[iMarker].pos.x(), data.normalized_marker[iMarker].pos.z()) * IMAGE_SIZE;

    const long annotationRow = long(imgPos.y() + 0.5);
    const long annotationCol = long(imgPos.x() + 0.5); //Round

    for (long rowOffset = -searchRadiusInPixel; rowOffset <= searchRadiusInPixel; rowOffset++)
    {
      for (long colOffset = -searchRadiusInPixel; colOffset <= searchRadiusInPixel; colOffset++)
      {
        const long imageRow = annotationRow + rowOffset;
        const long imageCol = annotationCol + colOffset;
        //Falls Position außerhalb des Bildes
        if (imageRow < 0 || imageRow >= IMAGE_SIZE || imageCol < 0 ||
          imageCol >= IMAGE_SIZE)
          continue;

        const Eigen::Vector2f pixelPos((double)imageCol, (double)imageRow);
        const float dist2 = (pixelPos - imgPos).squaredNorm();

        //außerhalb des Gaussians
        if (dist2 > dist2Threshold)
          continue;

        const long pixelIndex = imageRow * IMAGE_SIZE + imageCol;
        depthMap[pixelIndex] =
          std::max(data.normalized_marker[iMarker].pos.y(), depthMap[pixelIndex]);
      }
    }
  }
  data.image = depth_image(depthMap, IMAGE_SIZE, IMAGE_SIZE);
}

void ImageMethod::project() {
  Eigen::Matrix3f projectionMatrix = computeProjectionMatrix();
  projectedMarker.resize(marker.size());

  for (size_t iMarker = 0; iMarker < marker.size(); ++iMarker) {
     projectedMarker[iMarker].pos = projectionMatrix * marker[iMarker].pos;
     projectedMarker[iMarker].label = marker[iMarker].label;
  }
}

Eigen::Matrix3f ImageMethod::computeProjectionMatrix() {
  
  Eigen::Vector3f forward = principalAxis();
  Eigen::Vector3f dir = Eigen::Vector3f(0, 1, 0);

  //if y component of principalAxis is bigger, change to x direction to avoid sigularity
  if (abs(forward.y()) > abs(forward.x()))
    dir = Eigen::Vector3f(1, 0, 0);

  Eigen::Vector3f right = dir.cross(forward);
  Eigen::Vector3f up = forward.cross(right);

  Eigen::Matrix3f projection;
  projection(0, 0) = right.x();
  projection(0, 1) = right.y();
  projection(0, 2) = right.z();
  projection(1, 0) = forward.x();
  projection(1, 1) = forward.y();
  projection(1, 2) = forward.z();
  projection(2, 0) = up.x();
  projection(2, 1) = up.y();
  projection(2, 2) = up.z();

  return projection;
}

void ImageMethod::printImage(std::string& filename) {
  std::ofstream file;
  file.open(filename);
  file << "P2 " <<  IMAGE_SIZE << " " << IMAGE_SIZE << " 255\n";

  Eigen::MatrixXf imgValue = data.image.get_data();
  for (int i = 0; i < IMAGE_SIZE; ++i) {
    for (int j = 0; j < IMAGE_SIZE; ++j) {
      int val = int(imgValue(i, j) * 255 + 0.5);
      file << val << " ";
    }
    file << "\n";
  }
  file.close();
}