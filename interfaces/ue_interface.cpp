#include "ue_interface.h"
#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>

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


ue_interface::ue_interface() {

}

int ue_interface::init(const char* file) {
  LoadLibrary("torch_cuda.dll");
  if (!torch::cuda::is_available()) {
    std::cout << "Cuda not available\n";
    return 1;
  }
  //return 0;
  label = MarkerLabeling(file);
  return 0;
}

void ue_interface::readFile(const char* file) {
  size_t fileNumber = 0;
  size_t totalFileNumber = 1;

  std::vector<std::string> paths;
  //paths.push_back("D:/VR-PHI/OptiTracking/TrainingData/User5/capture1");
  paths.push_back(file);
  //SingleFile atm
  for (const auto& name : paths) {
    for (const auto& entry : std::filesystem::directory_iterator(name)) {
      if (fileNumber < totalFileNumber) {
        input = MarkerFileInput(entry.path().string());
        input.readFileLabels(); //read all frames in file
        ++fileNumber;
      }
    }
  }
}

void ue_interface::labelMarkers() {
  if (counter == 0) {
    marker = input.getNextFrame();
    image.setInputMarker(marker);
    inCNN data = image.createImage();
    label.PredictAll(data);
    previousFrame = label.GetMatched();
    label.SetLabels(marker);
    ik.setData(marker);
    //ik.DoIK();
    ++counter;
  }
  if (counter > 0 && counter < input.numberOfFrames()) {
    marker = input.getNextFrame();
    if (SetLabelsFromPrevious(previousFrame, marker)) {
      image.setInputMarker(marker);
      inCNN data = image.createImage();
      label.Predict(data);
      previousFrame = label.GetMatched();
      label.SetLabels(marker);
      ik.setData(marker);
    }
    ++counter;
  }
}

void ue_interface::getLabel() {
  if (counter < input.numberOfFrames()) {
    marker = input.getNextFrame();
    image.setInputMarker(marker);
    inCNN data = image.createImage();
    label.PredictAll(data);
    bPredictedLabel = label.IsLabelCorrect();
    ++counter;
  }
}

void ue_interface::inverseKinematic() {
  ik.setData(marker);
  ik.DoIK();
}

/*void ue_interface::readFile() {
  size_t fileNumber = 0;
  size_t totalFileNumber = 1;
  size_t totalFrameNumber = 0;
  ImagePCA image;

  std::vector<std::string> paths;
  paths.push_back("D:/VR-PHI/OptiTracking/TrainingData/User1/capture1");

  for (const auto& name : paths) {
    for (const auto& entry : std::filesystem::directory_iterator(name)) {
      if (fileNumber < totalFileNumber) {
        MarkerFileInput input(entry.path().string());
        input.readFile();
        auto frameNumber = input.numberOfFrames();
        size_t frames = 0;
        for (int j = 0; j < frameNumber; ++j) {
          std::vector<Marker> marker = input.getNextFrame();
          image.setInputMarker(marker);
          data.push_back(image.createImage());
          ++totalFrameNumber;
          ++frames;
        }
        ++fileNumber;
      }
    }
  }
}*/

void ue_interface::doLabeling() {
  size_t total = data.size();
  if (counter < total) {
    label.Predict(data[counter]);
    res = label.GetMarker();
    ++counter;
  }
}

void ue_interface::centerOnFinger() {
  std::vector <Eigen::Vector3f> marker = ik.GetMarkerData();
  ik.CenterAround(4);
  ik.FingerSpace((marker[4] - marker[5]).normalized());
  ik.RotateMarkers();

}

void ue_interface::getMarkerData(float* data, int nData) {
  std::vector <Eigen::Vector3f> marker = ik.GetMarkerData();
  for (int i = 0; i < nData; ++i) {
    data[3*i] = marker[i].x();
    data[3*i+1] = marker[i].y();
    data[3*i+2] = marker[i].z();
  }
}

void ue_interface::getNextMarkerDataLabels(float* data, bool* label, int nData) {
  //std::vector <Eigen::Vector3f> marker = ik.GetMarkerData();
  getLabel();
  sort();
  for (int i = 0; i < nData; ++i) {
    data[3 * i] = marker[i].pos.x();
    data[3 * i + 1] = marker[i].pos.y();
    data[3 * i + 2] = marker[i].pos.z();
    label[i] = bPredictedLabel[i];
  }
}

void ue_interface::getNormalizedPositions(float* data, int nData) {
  for (int i = 0; i < nData; ++i) {
    data[i] = 1;//res[i].pos.x();
    data[i + 1] = 2;//res[i].pos.y();
    data[i + 2] = 3;// res[i].pos.z();
  }
}

void ue_interface::sort() {
  label.SetLabels(marker);
  std::vector<Marker> tmp = marker;
  for (int i = 0; i < 19; ++i) {
    for (int j = 0; j < 19; ++j) {
      if (tmp[j].label == marker_label(i)) {
        marker[i] = tmp[j];
        continue;
      }
    }
  }
}


extern "C" __declspec(dllexport) iue_interface * __cdecl createTrackingObject()
{
  return new ue_interface;
}