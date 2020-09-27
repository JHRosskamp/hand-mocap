#pragma once
#include <stdexcept>
#include <ostream>
#include "Marker.h"


/// This enum class describes all available marker labels.

  // allow for printing a marker label to an output stream, e.g. stdout.
  std::ostream& operator<<(std::ostream& os, marker_label const& label);

  //marker_label from_training_data_label(std::string const &label_string);

  class bad_training_data_label : public std::runtime_error {
  private:
  public:
    bad_training_data_label(std::string const& label_string);
  };
