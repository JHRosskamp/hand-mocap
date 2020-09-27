#include <torch/torch.h>
#include <string>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <windows.h>
#include <chrono>
#include "MarkerFileInput.h"
#include "MarkerLabeling.h"
#include "ImagePCA.h"
#include "IK.h"

namespace fs = std::filesystem;
using namespace std;

bool SetLabelsFromPrevious(std::vector<Marker>& previousFrame, std::vector<Marker>& newFrame) {
  if (previousFrame.size() != newFrame.size())
    std::cout << "Ghost marker or missing marker" << std::endl;

  bool relabel = false;
  int index = 0;
  for (int i = 0; i < previousFrame.size(); ++i) {
    //std::cout << "Positions = " << previousFrame[i].filePosition << std::endl;
    //if not true then iterate previousframe until fileposition is again identical
    if (previousFrame[i].filePosition == newFrame[index].filePosition) {
      newFrame[index].label = previousFrame[i].label;
      ++index; // go to next element in newFrame.
    }
    else
      relabel = true;
  }
  return relabel;
}

void printLabels(std::vector<Marker>& frame) {
  std::cout << "Labels" << std::endl;
  for (auto& ele : frame) {
    std::cout << ele.label << "\t\twith\t\t" << ele.pos.x() << " " << ele.pos.y() << " " << ele.pos.z() << "\n";
  }
  std::cout << std::endl;
}

int main() {
  //Init torch
  LoadLibrary("torch_cuda.dll");
  if (!torch::cuda::is_available()) {
    std::cout << "Cuda not available\n";
    return 1;
  }
  MarkerLabeling label("model-pca.pt");// = MarkerLabeling("model-pca.pt");

  std::vector<std::string> paths;
  //paths.push_back("../TrainingData/Test");
  paths.push_back("../TrainingData/Calibration");

  ImagePCA image;
  IK ik;

  size_t fileNumber = 0;
  size_t totalFileNumber = 1000;
  size_t totalFrameNumber = 0;
  //Read File
  //label if necessary
  //ik

  for (const auto& name : paths) {
    for (const auto& entry : fs::directory_iterator(name)) {
      if (fileNumber < totalFileNumber) {
        std::cout << entry.path() << std::endl;
        MarkerFileInput input(entry.path().string());
        input.readFileLabels(); //read all frames in file
        auto frameNumber = input.numberOfFrames();
        //For first frame in file
        std::vector<Marker> marker = input.getNextFrame();
        image.setInputMarker(marker);
        inCNN data = image.createImage();
        label.PredictAll(data);
        std::vector<Marker> previousFrame = label.GetMatched();
        label.SetLabels(marker);
        //printLabels(previousFrame);
        printLabels(marker);
        ik.setData(marker);
        //ik.ConvertMmToCm();
        //====================================
        std::vector<Eigen::Vector3f> bla = ik.GetMarkerData();
        std::cout << "Test" << std::endl;
        for (int i = 0; i < 19; ++i)
          std::cout << i << "\t" << bla[i].x() << " " << bla[i].y() << " " << bla[i].z() << std::endl;
        //=======================

        //ik.DoIKRoot();
        ik.DoIKIndex();

        //===========
        bla = ik.GetMarkerData();
        std::cout << "Test" << std::endl;
        for(int i = 0; i<19; ++i)
          std::cout << i << "\t" << bla[i].x() << " " << bla[i].y() << " " << bla[i].z() << std::endl;
        //============

        //For all other frames
        /*size_t frames = 1;
        for (int j = 1; j < frameNumber; ++j) {
          marker = input.getNextFrame();
          std::cout << "Called new frame" << std::endl;
          if (SetLabelsFromPrevious(previousFrame, marker)) {
            //std::cout << "Relabeling" << std::endl;
            image.setInputMarker(marker);
            inCNN data = image.createImage();
            label.Predict(data);
            previousFrame = label.GetMatched();
            label.SetLabels(marker);
            //printLabels(previousFrame);
            //printLabels(marker);
          }
          ++frames; ++totalFrameNumber;
        }*/
        ++fileNumber;
      }

    }
  }
  return 0;
}