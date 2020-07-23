#include<iostream>
#include<filesystem>
#include "MarkerFileInput.h"
#include "ImagePCA.h"

namespace fs = std::filesystem;

int main() {
  //ImagePCA image;
  //image.setInputMarker();
  //inCNN data = image.createImage();
  size_t fileNumber = 1;

  std::vector<std::string> paths;
  paths.push_back("../TrainingData/User1/capture1");
  for (const auto& name : paths) {
    for (const auto& entry : fs::directory_iterator(name)) {
      for (int i = 0; i < fileNumber; ++i) {
        MarkerFileInput input(entry.path().string());
        auto frameNumber = input.numberOfFrames();
        for (int j = 0; j < frameNumber; ++j) {
          std::vector<Marker> marker = input.getNextFrame();
        }
        ++fileNumber;
      }
      
    }
  }


  return 0;
}