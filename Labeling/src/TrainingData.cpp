#include<iostream>
#include<filesystem>
#include<fstream>
#include<string>
#include "MarkerFileInput.h"
#include "ImagePCA.h"

namespace fs = std::filesystem;
using namespace std;

void saveDepthImage(vector<inCNN>& data) {
  string filename = "TrainingImage.dat";
  ofstream file;

  file.open(filename);
  for (size_t i = 0; i < data.size(); ++i) {
    Eigen::MatrixXf img = data[i].image.get_data();
    int col = img.cols();
    int row = img.rows();
    for (int rows = 0; rows < row; ++rows) {
      for (int cols = 0; cols < col; ++cols) {
        file << img(rows, cols) << "\t";
      }
    }
    file << "\n";
  }
  file.close();
}

void saveNormalizedMarker(vector<inCNN>& data) {
  string filename = "TrainingMarker.dat";
  ofstream file;

  file.open(filename);
  for (size_t i = 0; i < data.size(); ++i) {
    for (size_t j = 0; j < data[i].normalized_marker.size(); ++j) {
      file << data[i].normalized_marker[j].pos.x() << "\t" <<
              data[i].normalized_marker[j].pos.y() << "\t" <<
              data[i].normalized_marker[j].pos.z() << "\t";
   
    }
    file << "\n";
  }
  file.close();
}

int main() {
  ImagePCA image;
  std::vector<inCNN> data;
  size_t fileNumber = 0;
  size_t totalFileNumber = 1;

  std::vector<std::string> paths;
  paths.push_back("../TrainingData/User1/capture1");
  for (const auto& name : paths) {
    for (const auto& entry : fs::directory_iterator(name)) {
      while(fileNumber < totalFileNumber) {
        MarkerFileInput input(entry.path().string());
        input.readFile();
        auto frameNumber = input.numberOfFrames();
        for (int j = 0; j < frameNumber; ++j) {
          std::vector<Marker> marker = input.getNextFrame();
          image.setInputMarker(marker);
          data.push_back(image.createImage());
        }
        ++fileNumber;
      }
      
    }
  }
  saveNormalizedMarker(data);
  saveDepthImage(data);

  return 0;
}