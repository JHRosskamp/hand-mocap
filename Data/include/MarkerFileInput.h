#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include "Marker.h"
#include "Metadata.h"

class MarkerFileInput {
public:
  MarkerFileInput(std::string const& filename);

  std::vector<Marker> getNextFrame() {
     ++counter;
     return data[counter];
  }

  metadata getMetadata() {
    return metadata;
  }

  int numberOfFrames () const {
    return frameNumber;
  }
  void readFile();
  std::vector<std::string> splitString(std::string s);
private:
  //std::string filename;
  void readMarker();
  void readMetadata();

  std::ifstream file;
  metadata metadata;
  std::vector<std::vector<Marker>> data;
  //std::vector<
  int counter = -1;
  int frameNumber = 0;
};


