#include <torch/torch.h>
#include <string>
#include <fstream>
#include <Eigen/Dense>
#include "MarkerFileInput.h"

constexpr int imageSize = 52;

class BaseCNN {
protected:
  std::vector<torch::Tensor> processImages(std::vector<std::string> listImages) {
    std::vector<torch::Tensor> imageVector;
    for (auto& it : listImages) {
      torch::Tensor image = readImages(it);
      imageVector.push_back(image);
    }
    return imageVector;
  }

  std::vector<torch::Tensor> processMarkers(std::vector<std::string> listMarkers) {
    std::vector<torch::Tensor> markerVector;
    for (auto& it : listMarkers) {
      torch::Tensor marker = readMarkers(it);
      markerVector.push_back(marker);
    }
    return markerVector;
  }

  torch::Tensor readImages(std::string path) {
    std::ifstream file;
    std::string line;
    file.open(path);
    
    auto options = torch::TensorOptions().device(torch::kCUDA);
    float mat[imageSize*imageSize];
    int j = 0;
    while (std::getline(file, line)) {

      std::vector<std::string> elements = splitString(line);
      for (int i = 0; i < imageSize; ++i) {
        float val = stof(elements[i]);
        mat[i + j * imageSize] = val;
      }
    }

    torch::Tensor imageTensor = torch::from_blob(mat, { imageSize,imageSize,1 }, options);
    imageTensor = imageTensor.permute({ 2, 0, 1 }); // Channels x Height x Width
    return imageTensor.clone();
  }

  torch::Tensor readMarkers(std::string path) {
    std::ifstream file;
    std::string line;
    file.open(path);

    auto options = torch::TensorOptions().device(torch::kCUDA);
    float mat[19 * 3];
    int j = 0;
    while (std::getline(file, line)) {

      std::vector<std::string> elements = splitString(line);
      for (int i = 0; i < 3; ++i) {
        float val = stof(elements[i]);
        mat[i + j * 3] = val;
      }
    }

    torch::Tensor imageTensor = torch::from_blob(mat, { imageSize,imageSize }, options);
    //imageTensor = imageTensor.permute({ 2, 0, 1 }); // Channels x Height x Width
    return imageTensor.clone();
  }
};

class CustomDataset : public torch::data::Dataset<CustomDataset> {

};