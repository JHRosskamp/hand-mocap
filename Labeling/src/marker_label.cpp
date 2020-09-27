#include "marker_label.h"

#include <unordered_map>
#include <string>

//#include <boost/algorithm/string.hpp>

    namespace {
        std::unordered_map<std::string, marker_label> mapping{
            {"ThumbCMC", marker_label::thumb_base1},
            {"ThumbMCP", marker_label::thumb_base2},
            {"ThumbIP", marker_label::thumb_middle},
            {"ThumbTip", marker_label::thumb_tip},
            {"IndexMCP", marker_label::index_knuckle},
            {"IndexPIP", marker_label::index_middle},
            {"IndexTip", marker_label::index_tip},
            {"MiddleMCP", marker_label::middle_knuckle},
            {"MiddlePIP", marker_label::middle_middle},
            {"MiddleTip", marker_label::middle_tip},
            {"RingMCP", marker_label::ring_knuckle},
            {"RingPIP", marker_label::ring_middle},
            {"RingTip", marker_label::ring_tip},
            {"PinkyMCP", marker_label::little_knuckle},
            {"PinkyPIP", marker_label::little_middle},
            {"PinkyTip", marker_label::little_tip},
            {"BackHand1", marker_label::hand_base_little},
            {"BackHand2", marker_label::hand_base_wrist},
            {"BackHand3", marker_label::hand_base_thumb},
        };
    }

    std::ostream& operator<<(std::ostream &os, marker_label const &label) {
        switch (label) {
            case marker_label::thumb_base1:
                os << "thumb_base1";
                break;
            case marker_label::thumb_base2:
                os << "thumb_base2";
                break;
            case marker_label::thumb_middle:
                os << "thumb_middle";
                break;
            case marker_label::thumb_tip:
                os << "thumb_tip";
                break;
            case marker_label::index_knuckle:
                os << "index_knuckle";
                break;
            case marker_label::index_middle:
                os << "index_middle";
                break;
            case marker_label::index_tip:
                os << "index_tip";
                break;
            case marker_label::middle_knuckle:
                os << "middle_knuckle";
                break;
            case marker_label::middle_middle:
                os << "middle_middle";
                break;
            case marker_label::middle_tip:
                os << "middle_tip";
                break;
            case marker_label::ring_knuckle:
                os << "ring_knuckle";
                break;
            case marker_label::ring_middle:
                os << "ring_middle";
                break;
            case marker_label::ring_tip:
                os << "ring_tip";
                break;
            case marker_label::little_knuckle:
                os << "little_knuckle";
                break;
            case marker_label::little_middle:
                os << "little_middle";
                break;
            case marker_label::little_tip:
                os << "little_tip";
                break;
            case marker_label::hand_base_little:
                os << "hand_base_little";
                break;
            case marker_label::hand_base_wrist:
                os << "hand_base_wrist";
                break;
            case marker_label::hand_base_thumb:
                os << "hand_base_thumb";
                break;
            case marker_label::no_label:
                os << "no_label";
                break;
            default:
                throw std::runtime_error{"unknown marker label: " + std::to_string(static_cast<int>(label))};
        }
        return os;
    }

    /*marker_label from_training_data_label(std::string const &label_string) {
        std::string trimmed_label_string{label_string};
        boost::trim(trimmed_label_string);
        
        try {
            return mapping.at(trimmed_label_string);
        } catch (std::out_of_range const &) {
            throw bad_training_data_label{trimmed_label_string};
        }
    }*/

    bad_training_data_label::bad_training_data_label(std::string const &label_string)
            : std::runtime_error{"bad marker label in training data: " + label_string} {
    }
