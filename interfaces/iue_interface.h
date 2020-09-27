#pragma once

class iue_interface {
public:
  virtual int init() = 0;
  virtual void readFile() = 0;
  virtual void labelMarkers() = 0;
  virtual void inverseKinematic() = 0;
  virtual void doLabeling() = 0;
  virtual void centerOnFinger() = 0;
  virtual void getNormalizedPositions(float* data, int nData) = 0;
  virtual void getMarkerData(float* data, int nData) = 0;
  virtual void getNextMarkerDataLabels(float* data, bool* label, int nData) = 0;
};