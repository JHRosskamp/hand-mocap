#ifndef HAND_TRACKING_DEEP_LABELING_MARKER_MATCHING_H
#define HAND_TRACKING_DEEP_LABELING_MARKER_MATCHING_H

#include <vector>
#include <Eigen/Dense>
#include "marker_label.h"

    /// The result / solution of an instance of the weighted bipartite matching problem.
struct matching_result {
        /// The matched labels in order of the input, i.e. the first element of this
        /// vector contains the label for the first marker in the input vector.
        std::vector<marker_label> labels;
        /// The total cost of the matching / flow of the underlying minimum-cost flow problem.
        float total_matching_cost;
};

    /// Solves the weighted bipartite matching problem by representing it as a minimum-cost flow
    /// problem and then solving this network flow problem.
    ///
    /// \param normalized_markers The original input markers, but rotated to adjust to the orientation of
    ///        the network output.
    /// \param prediction The network output adjusted to match the axis orientation of the input,
    ///                            centered around the origin and scaled based on the XY-extent of all
    ///                            rotated input markers.
    /// \return The solution of the problem and the resulting cost of the matching.
matching_result solve_matching_problem(std::vector<Eigen::Vector3f> const &normalized_markers,
                                           std::vector<Eigen::Vector3f> const &prediction);

#endif //HAND_TRACKING_DEEP_LABELING_MARKER_MATCHING_H
