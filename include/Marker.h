#pragma once
#include <vector>
#include <Eigen/Dense>

enum class marker_label : int {
  thumb_base1 = 0,
  thumb_base2,
  thumb_middle,
  thumb_tip,

  index_knuckle,
  index_middle,
  index_tip,

  middle_knuckle,
  middle_middle,
  middle_tip,

  ring_knuckle,
  ring_middle,
  ring_tip,

  little_knuckle,
  little_middle,
  little_tip,

  hand_base_little,
  hand_base_wrist,
  hand_base_thumb,

  no_label,
};

class depth_image {
public:
  depth_image() {};

  depth_image(Eigen::MatrixXf data) {
    this->data = data;
  };

  depth_image(std::vector<float> val, int sizex, int sizey) {
    data.resize(sizex, sizey);
    for (int i = 0; i < sizex; ++i)
    {
      for (int j = 0; j < sizey; ++j)
      {
        int index = i * sizey + j;
        data(i, j) = val[index];
      }
    }
  };

  void get_in_row_major_order() {
  };

  Eigen::MatrixXf get_data() {
    return data;
  };

  Eigen::MatrixXf const& matrix() const { return this->data; };

private:
  Eigen::MatrixXf data;
};

class Marker {
public:
  Marker() {};

  Marker(Eigen::Vector3f pos, marker_label label) {
    this->pos = pos;
    this->label = label;
  }

  Eigen::Vector3f pos;
  marker_label label;
};

class inCNN {
public:
  std::vector<Marker> normalized_marker;
  depth_image image;
};