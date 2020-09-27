#pragma once

#include <torch/torch.h>
#include <string>
#include <fstream>
#include <Eigen/Dense>
#include <exception>
#include "MarkerFileInput.h"

constexpr int imageSize = 52;
constexpr int numberMarker = 19;
torch::Device device(torch::kCUDA);

torch::Tensor readImage(depth_image& img) {
  Eigen::MatrixXf image = img.get_data();
  float mat[imageSize * imageSize];

  for (int rows = 0; rows < imageSize; ++rows) {
      for (int cols = 0; cols < imageSize; ++cols) {
          mat[ cols + rows * imageSize] = image(rows, cols);
      }
  }
  torch::Tensor imageTensor = torch::from_blob(mat, { imageSize,imageSize,1 });
  imageTensor = imageTensor.permute({ 2, 0, 1 });
  return imageTensor.clone();
}

torch::Tensor readMarker(std::vector<Marker>& marker) {
  float mat[numberMarker * 3];
  for (int j = 0; j < numberMarker; ++j) {
      Eigen::Vector3f pos = marker[j].pos;
      mat[0 + j * 3] = pos.x();
      mat[1 + j * 3] = pos.y();
      mat[2 + j * 3] = pos.z();    
  }

  torch::Tensor imageTensor = torch::from_blob(mat, { 19,3 });
  return imageTensor.clone();
}

torch::Tensor readImages(std::string path) {
  std::ifstream file;
  std::string line;
  file.open(path);
  auto options = torch::TensorOptions().device(torch::kCUDA);
  float mat[imageSize * imageSize];
  int j = 0;
  while (std::getline(file, line)) {

    std::vector<std::string> elements = splitString(line);
    for (int i = 0; i < imageSize; ++i) {
      float val = stof(elements[i]);
      mat[i + j * imageSize] = val;
    }
    ++j;
  }

  torch::Tensor imageTensor = torch::from_blob(mat, { imageSize,imageSize,1 });
  imageTensor = imageTensor.permute({ 2, 0, 1 }); // Channels x Height x Width
  return imageTensor.clone();
}

torch::Tensor readMarkers(std::string path) {
  std::ifstream file;
  std::string line;
  file.open(path);

  auto options = torch::TensorOptions().device(torch::kCUDA);
  float mat[numberMarker * 3];
  int j = 0;
  while (std::getline(file, line)) {

    std::vector<std::string> elements = splitString(line);
    for (int i = 0; i < 3; ++i) {
      float val = stof(elements[i]);
      mat[i + j * 3] = val;
    }
    ++j;
  }
  //std::cout << "Before copy\n";
  torch::Tensor imageTensor = torch::from_blob(mat, { 19,3 });
  //imageTensor.to(at::kCUDA);
  //std::cout << "after copy\n";
  //imageTensor = imageTensor.permute({ 2, 0, 1 }); // Channels x Height x Width
  return imageTensor.clone();
}

std::vector<torch::Tensor> processImages(std::vector<std::string> listImages) {
  std::vector<torch::Tensor> imageVector;
  for (auto& it : listImages) {
    torch::Tensor image = readImages(it).to(device);// .to(at::kCUDA);
    imageVector.push_back(image);
  }
  return imageVector;
}

std::vector<torch::Tensor> processMarkers(std::vector<std::string> listMarkers) {
  std::vector<torch::Tensor> markerVector;
  for (auto& it : listMarkers) {
    torch::Tensor marker = readMarkers(it).to(device);// .to(at::kCUDA);
    markerVector.push_back(marker);
  }
  return markerVector;
}

class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
  std::vector<torch::Tensor> images, markers;
public:
  CustomDataset(std::vector<std::string> listImages, std::vector<std::string> listMarkers) {
    images = processImages(listImages);
    markers = processMarkers(listMarkers);
  };

  torch::data::Example<> get(size_t index) override {
    torch::Tensor sample_img = images.at(index);
    torch::Tensor sample_markers = markers.at(index);
    return { sample_img.clone(), sample_markers.clone() };
  };

  // Return the length of data
  torch::optional<size_t> size() const override {
    return markers.size();
  };
};