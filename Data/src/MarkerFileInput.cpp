#include "MarkerFileInput.h"
#include <filesystem>


std::vector<std::string> splitString(std::string s) {
  size_t pos = 0;
  std::vector<std::string> element;
  std::string delimiter = "\t";
  while ((pos = s.find(delimiter)) != std::string::npos) {
    element.push_back(s.substr(0, pos));
    s.erase(0, pos + delimiter.length());
  }
  element.push_back(s);
  return element;
}

MarkerFileInput::MarkerFileInput(const std::string& filename) {
  std::filesystem::path targetFile(filename);

  if (!std::filesystem::is_regular_file(targetFile))
    std::cout << "Path does not point to file" << std::endl;
  
  file.open(filename);
  if (!file)
    std::cout << "Failed to open file" << std::endl;

  //getMetadata();
  //readFile();
}

void MarkerFileInput::readFile() {
  readMetadata();
  readMarker();
}

void MarkerFileInput::readFileLabels() {
  readMetadata();
  readMarkerLabels();
}

void MarkerFileInput::readMarker() {
  std::string line;

  while (std::getline(file, line)) {
    std::vector<Marker> frame;
    auto lineElements = splitString(line);
    auto const expected_length = metadata.num_markers * 3 + 2;
    if (lineElements.size() == expected_length + 1) 
      // the tracking data from the paper contains an additional tab character
      // after the last coordinate, which produces an extra empty element.
      lineElements.pop_back();
    for (int i = 2, labelID = 0; i < lineElements.size(); i+=3, ++labelID) {
      if (lineElements[i].empty()) //should not happen if training data.
        continue;
      Marker marker;
      marker.pos = Eigen::Vector3f(std::stof(lineElements[i]),
                                     std::stof(lineElements[i+1]),
                                     std::stof(lineElements[i+2]));
      //TODO only if training data. otherwise undefined
      marker.label = marker_label(labelID); 

      frame.push_back(marker);
    }
    data.push_back(frame);
  }
  frameNumber = data.size();
}

//Ignores frames with !=19 markers
void MarkerFileInput::readMarkerLabels() {
  std::string line;

  while (std::getline(file, line)) {
    std::vector<Marker> frame;
    auto lineElements = splitString(line);
    auto const expected_length = metadata.num_markers * 3 + 2;
    if (lineElements.size() == expected_length + 1) //WARNING
      // the tracking data from the paper contains an additional tab character
      // after the last coordinate, which produces an extra empty element. Here not needed probably.
      lineElements.pop_back();

    for (int i = 2, labelID = 0; i < lineElements.size(); i += 3, ++labelID) {
      if (lineElements[i].empty()) { //in case marker is missing
        continue;
      }
      Marker marker;
      marker.pos = Eigen::Vector3f(std::stof(lineElements[i]),
                                   std::stof(lineElements[i + 1]),
                                   std::stof(lineElements[i + 2]));
      //undefined
      marker.label = marker_label::no_label;
      marker.filePosition = i;

      frame.push_back(marker);
    }
    //check if every marker is visible
    if(frame.size() == 19)
      data.push_back(frame);
  }
  frameNumber = data.size();
}

void MarkerFileInput::readMetadata() {
  std::string line;
  //unimportant
  std::getline(file, line);
  std::getline(file, line);

  std::getline(file, line);
  auto const lineElements = splitString(line);
 
  metadata.data_rate = std::stoi(lineElements[0]);
  metadata.camera_rate = std::stoi(lineElements[1]);
  metadata.num_frames = std::stoi(lineElements[2]);
  metadata.num_markers = std::stoi(lineElements[3]);
  metadata.units = lineElements[4];
  metadata.orig_data_rate = std::stoi(lineElements[5]);
  metadata.orig_data_start_frame = std::stoi(lineElements[6]);
  metadata.orig_num_frames = std::stoi(lineElements[7]);
  
  std::getline(file, line);
  //Labels if order changed
  std::getline(file, line);
}

