#pragma once

#include "iue_interface.h"
#include "MarkerFileInput.h"
#include "MarkerLabeling.h"
#include "ImagePCA.h"
#include "IK.h"
#include <torch/torch.h>
#include <windows.h>

class ue_interface : public iue_interface {
public:
  ue_interface();
  int init();
  void readFile();
  void doLabeling();
  void centerOnFinger();
  void labelMarkers();
  void inverseKinematic();
  void getNormalizedPositions(float* data, int nProj);
  void getMarkerData(float* data, int nData);
  void getNextMarkerDataLabels(float* data, bool* label, int nData);

private:
  void getLabel();
  void sort();

  std::vector<inCNN> data;
  MarkerLabeling label;
  MarkerFileInput input;
  size_t frameNumber;
  size_t counter = 0;
  std::vector<Marker> res;
  ImagePCA image;
  IK ik;
  std::vector<Marker> previousFrame;
  std::vector<Marker> marker;
  std::vector<bool> bPredictedLabel;
  //VGG8 net;
};