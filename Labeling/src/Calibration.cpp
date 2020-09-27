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

void printLabels(std::vector<Marker>& frame) {
  std::cout << "Labels" << std::endl;
  for (auto& ele : frame) {
    std::cout << ele.label << "\t\twith\t\t" << ele.pos.x() << " " << ele.pos.y() << " " << ele.pos.z() << "\n";
  }
  std::cout << std::endl;
}

void printLabels(std::vector<Eigen::Vector3f>& frame) {
  std::cout << "Labels" << std::endl;
  for (int i = 0; i < frame.size(); ++i) {
    if( i == 4 || i==5 || i==6 || i==16 || i==17 || i==18)
    std::cout << marker_label(i) << "\t\twith\t\t" << frame[i].x() << " " << frame[i].y() << " " << frame[i].z() << "\n";
  }
  std::cout << std::endl;
}

int main() {
  LoadLibrary("torch_cuda.dll");
  if (!torch::cuda::is_available()) {
    std::cout << "Cuda not available\n";
    return 1;
  }
  ImagePCA image;
  IK ik;

  MarkerLabeling label("model-pca.pt");// = MarkerLabeling("model-pca.pt");
  //label.

  size_t fileNumber = 0;
  size_t totalFileNumber = 1;
  size_t totalFrameNumber = 0;

  std::vector<std::string> paths;
  paths.push_back("../TrainingData/Calibration");
  for (const auto& name : paths) {
    for (const auto& entry : fs::directory_iterator(name)) {
      if (fileNumber < totalFileNumber) {
        MarkerFileInput input(entry.path().string());
        input.readFile();
        auto frameNumber = 1;// input.numberOfFrames();
        size_t frames = 0;
        for (int i = 0; i < 30; ++i) {
          std::vector<Marker> marker = input.getNextFrame(); //ignore first line
        }
        for (int j = 0; j < frameNumber; ++j) {
          std::vector<Marker> marker = input.getNextFrame();
          image.setInputMarker(marker);
          inCNN data = image.createImage(); //IK
          label.PredictAll(data);
          label.SetLabels(marker);
          ik.setData(marker);
          //std::vector<Eigen::Vector3f> res = ik.GetMarkerData();
          //printLabels(res);
          //float dist1 = (res[4] - res[6]).norm();
          std::vector<Eigen::Vector3f> res = ik.GetMarkerData();
          printLabels(res);
          ik.CenterAround(16);
          //set x-axis and get transform
          //ik.FingerSpace((res[4] - res[6]).normalized());
          //ik.RotateMarkers();
          //ik.ProjectMarkers();
          //ik.DoIKForward();
          Eigen::VectorXd ikres = ik.DoIKRoot();
          ik.DoIKForward(ikres);
          res = ik.GetMarkerData();
          printLabels(res);
          float dist2 = (res[4] - res[6]).norm();



          //printLabels(res);
          ++totalFrameNumber;
          ++frames;
        }
        ++fileNumber;
      }

    }
  }
  return 0;
}