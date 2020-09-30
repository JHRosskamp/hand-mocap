#include<iostream>
#include<filesystem>
#include<fstream>
#include<string>
#include<sstream>
#include "MarkerFileInput.h"
#include "ImagePCA.h"
#include "ImagePalm.h"
#include "ImageRandom.h"

namespace fs = std::filesystem;
using namespace std;

void saveDepthImage(inCNN& data, size_t frame) {
  stringstream filename;
  ofstream file;
  filename << "ImageData\\TrainingImage" << frame << ".dat";
  
  file.open(filename.str());
  Eigen::MatrixXf img = data.image.get_data();
  int col = img.cols();
  int row = img.rows();
  for (int rows = 0; rows < row; ++rows) {
    for (int cols = 0; cols < col; ++cols) {
      file << img(rows, cols) << "\t";
    }
    file << "\n";
  }

  file.close();
}

void saveDepthImageBatch(std::vector<inCNN>& data, size_t frame) {
  stringstream filename;
  ofstream file;
  filename << "ImageDataBatch\\TrainingImage" << frame << ".dat";

  file.open(filename.str());
  for (int i = 0; i < data.size(); ++i) {
    Eigen::MatrixXf img = data[i].image.get_data();
    int col = img.cols();
      int row = img.rows();
      for (int rows = 0; rows < row; ++rows) {
          for (int cols = 0; cols < col; ++cols) {
              file << img(rows, cols) << "\t";
          }
          file << "\n";
      }
      file << "\n";
  }
  file.close();
}


void saveNormalizedMarker(inCNN& data, size_t frame) {
  stringstream filename;
  ofstream file;
  filename << "MarkerData\\TrainingMarker" << frame << ".dat";

  file.open(filename.str());
  for (size_t j = 0; j < data.normalized_marker.size(); ++j) {
    file << data.normalized_marker[j].pos.x() << "\t" <<
            data.normalized_marker[j].pos.y() << "\t" <<
            data.normalized_marker[j].pos.z() << "\t";
    file << "\n";
  }

  file.close();
}

void saveNormalizedMarkerBatch(std::vector<inCNN>& data, size_t frame) {
  stringstream filename;
  ofstream file;
  filename << "MarkerDataBatch\\TrainingMarker" << frame << ".dat";

  file.open(filename.str());
  for (int i = 0; i < data.size(); ++i) {
    for (size_t j = 0; j < data[i].normalized_marker.size(); ++j) {
       file << data[i].normalized_marker[j].pos.x() << "\t" <<
               data[i].normalized_marker[j].pos.y() << "\t" <<
               data[i].normalized_marker[j].pos.z() << "\t";
       file << "\n";
    }
    file << "\n";
  }
  file.close();
}

int main() {
  ImagePCA image;
  //ImagePalm image;
  //ImageRandom image;
  size_t fileNumber = 0;
  size_t totalFileNumber = 1000;
  size_t totalFrameNumber = 0;
  size_t filesSaved = 0;

  std::vector<std::string> paths;
  paths.push_back("../TrainingData/User1/capture1");
  paths.push_back("../TrainingData/User1/capture2");
  paths.push_back("../TrainingData/User2/capture1");
  paths.push_back("../TrainingData/User2/capture2");
  paths.push_back("../TrainingData/User2/capture3");
  paths.push_back("../TrainingData/User2/capture4");
  paths.push_back("../TrainingData/User3/capture1");
  paths.push_back("../TrainingData/User3/capture2");
  paths.push_back("../TrainingData/User3/capture3");
  paths.push_back("../TrainingData/User3/capture4");
  paths.push_back("../TrainingData/User4/capture1");
  paths.push_back("../TrainingData/User4/capture2");
  //paths.push_back("../TrainingData/User5/capture1");
  for (const auto& name : paths) {
    ++filesSaved;
    std::vector<inCNN> data_vec;
    for (const auto& entry : fs::directory_iterator(name)) {
      if(fileNumber < totalFileNumber) {
        std::cout << entry.path() << std::endl;
        MarkerFileInput input(entry.path().string());
        input.readFile();
        auto frameNumber = input.numberOfFrames();
 
        for (int j = 0; j < frameNumber; ++j) {
          std::vector<Marker> marker = input.getNextFrame();
          image.setInputMarker(marker);
          inCNN data = image.createImage();
          data_vec.push_back(data);
          ++totalFrameNumber;
        }
        ++fileNumber;
      }
      
    }
    saveNormalizedMarkerBatch(data_vec, filesSaved);
    saveDepthImageBatch(data_vec, filesSaved);
  }


  return 0;
}
