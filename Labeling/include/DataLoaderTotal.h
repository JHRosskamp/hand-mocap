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


std::vector<torch::Tensor> readImagesBatch(std::string path) {
  std::vector<torch::Tensor> res;
  std::ifstream file;
  std::string line;
  file.open(path);
  auto options = torch::TensorOptions();// .device(torch::kCUDA);
  float mat[imageSize * imageSize];
  int j = 0;
  while (std::getline(file, line)) {
    if (line.empty()) {
      j = 0;
      continue;
    }
    std::vector<std::string> elements = splitString(line);
    for (int i = 0; i < imageSize; ++i) {
      float val = stof(elements[i]);
      mat[i + j * imageSize] = val;
    }
    if (j == 51) {
      torch::Tensor imageTensor = torch::from_blob(mat, { imageSize,imageSize,1 });
      imageTensor = imageTensor.permute({ 2, 0, 1 }).to(device); // Channels x Height x Width
      res.push_back(imageTensor.clone());
    }
    ++j;
  }
  return res;
}

std::vector<torch::Tensor> readMarkersBatch(std::string path) {
  std::vector<torch::Tensor> res;
  std::ifstream file;
  std::string line;
  file.open(path);

  float mat[numberMarker * 3];
  int j = 0;
  while (std::getline(file, line)) {
    if (line.empty()) {
      j = 0;
      continue;
    }
    std::vector<std::string> elements = splitString(line);
    for (int i = 0; i < 3; ++i) {
      float val = stof(elements[i]);
      mat[i + j * 3] = val;
    }
    if (j == 18) {
      torch::Tensor imageTensor = torch::from_blob(mat, { 19,3 }).to(device);
      res.push_back(imageTensor.clone());
    }
    ++j;
  }
  return res;
}

std::vector<torch::Tensor> processImages(std::vector<std::string> listImages) {
  std::vector<torch::Tensor> imageVector;
  for (auto& it : listImages) {
    std::vector<torch::Tensor> tmp;
    tmp = readImagesBatch(it);
    imageVector.insert(imageVector.end(), tmp.begin(), tmp.end());
  }
  return imageVector;
}

std::vector<torch::Tensor> processMarkers(std::vector<std::string> listMarkers) {
  std::vector<torch::Tensor> markerVector;
  for (auto& it : listMarkers) {
    std::vector<torch::Tensor> tmp;
   tmp = readMarkersBatch(it);// .to(at::kCUDA);
   markerVector.insert(markerVector.end(), tmp.begin(), tmp.end());
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