#pragma once
/// tracking_data contains basic information found in the header of a trc-file.
struct metadata {

  //std::string initial_export_location;
  unsigned int data_rate, camera_rate, num_frames, num_markers, orig_data_rate, orig_data_start_frame, orig_num_frames;
  std::string units;

  metadata() : data_rate(0), camera_rate(0),
    num_frames(0), num_markers(0),
    orig_data_rate(0),
    orig_data_start_frame(0),
    orig_num_frames(0) {};
};