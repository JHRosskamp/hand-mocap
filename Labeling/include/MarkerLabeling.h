#pragma once

#include <string>
#include "Marker.h"
//#include "DataLoader.h"
#include "VGG8.h"
#include "marker_matching.h"
#include "marker_label.h"
#include <torch/torch.h>

class MarkerLabeling
{
public:
  MarkerLabeling(std::string model);
  MarkerLabeling();

  void SetLabelingMethod();

  void Predict(inCNN& data);
  void PredictAll(inCNN& data);
  std::vector<Marker> GetMarker();
  std::vector<Marker> GetMatched();
  std::vector<bool> IsLabelCorrect() const { return bPredictedLabel; };
  void SetLabels(std::vector<Marker>& data);
  void PrintAccuracy();

private:
  void CallCNN();
  void Matching();
  void MatchingAll();
  torch::Tensor readImage(depth_image& img);

  inCNN input;
  std::vector<Marker> outMarker;
  std::vector<Marker> labels;
  std::vector<bool> bPredictedLabel;
  VGG8 net;

  size_t right = 0; size_t wrong = 0;


};