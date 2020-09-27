#include "marker_matching.h"

#include <string>

// lemon/maps.h has to be included first, it is missing in capacity_scaling.h:
// https://lemon.cs.elte.hu/trac/lemon/ticket/600
#include <lemon/maps.h>
#include <lemon/capacity_scaling.h>
#include <lemon/list_graph.h>
#include <lemon/smart_graph.h>
#include <lemon/lgf_writer.h>
#include <marker_label.h>

matching_result solve_matching_problem(std::vector<Eigen::Vector3f> const &normalized_markers,
                                           std::vector<Eigen::Vector3f> const &prediction) {
        /*if (normalized_markers.size() < 19) {
            throw std::runtime_error{"the number of the original input markers (normalized) is below 19: "
                                     + std::to_string(normalized_markers.size())};
        }

        if (prediction.size() != 19) {
            throw std::runtime_error{"the length of network output is not 19: "
                                     + std::to_string(prediction.size())};
        }*/

        // sources weighted bipartite matching:
        // Steven S. Skiena: The Algorithm Design Manual
        //   pp. 217: basic bipartite matching + representation as network flow problem
        //   pp. 498: (weighted) matching problem on graphs (which can be bipartite)
        //   pp. 509: network flow + minimum cost flow
        //
        // Cormen et. al.: Introduction to Algorithms
        //   pp. 843: introduction to LPs
        //   pp. 861: minimum cost flow LP example
        // note: LPs are not used here, this references are here only to provide an alternative
        //       solution for the interested reader
        //
        // Lemon library documentation
        //   - provides solver for minimum cost flow
        //   - http://lemon.cs.elte.hu/pub/doc/1.3.1/a00612.html
        //
        // https://en.wikipedia.org/wiki/Minimum-cost_flow_problem#Minimum_weight_bipartite_matching
        // is based on:
        // => simplified minimum cost flow for weighted bipartite matching only using supply
        //    on the left (all 1) and right (all -1) nodes: R. Ahuja et al (1993): Network Flows
        // further reading:
        // => see Skiena p.511 for references to other sources about network flows

        lemon::SmartDigraph graph;

        // arc: directed edge
        auto const source = graph.addNode();
        auto const sink = graph.addNode();

        decltype(graph)::ArcMap<float> costs{graph};
        decltype(graph)::ArcMap<float> lower_bounds{graph};

        // Arc cost is L2-norm of diff between observed marker (adjusted)
        //
        // constructing graph:
        // - with source having an arc to each normalized marker (the `left` side) (lower bound = 0, cost = 0)
        // - with each predicted marker (the `right` side) having an arc to the sink (lower bound = 1, cost = 0)
        // - with each normalized marker having an arc to each predicted marker (lower bound = 0, cost = L2 of diff)
        // - with each arc having the capacity of 1 to prevent arc reuse
        std::vector<decltype(graph)::Node> normalized_nodes;
        for (auto normalized_idx = 0ul; normalized_idx < normalized_markers.size(); normalized_idx++) {
            auto const normalized_node = graph.addNode();
            normalized_nodes.push_back(normalized_node);

            auto const source_to_normalized = graph.addArc(source, normalized_node);
            lower_bounds[source_to_normalized] = 0;
            costs[source_to_normalized] = 0;
        }

        std::vector<decltype(graph)::Node> predicted_nodes;
        for (auto predicted_idx = 0ul; predicted_idx < prediction.size(); predicted_idx++) {
            auto const predicted_node = graph.addNode();
            predicted_nodes.push_back(predicted_node);

            auto const predicted_to_sink = graph.addArc(predicted_node, sink);
            lower_bounds[predicted_to_sink] = 1;
            costs[predicted_to_sink] = 0;
        }

        std::vector<decltype(graph)::Arc> arcs;
        for (auto normalized_idx = 0ul; normalized_idx < normalized_markers.size(); normalized_idx++) {
            for (auto predicted_idx = 0ul; predicted_idx < prediction.size(); predicted_idx++) {
                auto const normalized_to_predicted =
                        graph.addArc(normalized_nodes[normalized_idx], predicted_nodes[predicted_idx]);
                arcs.push_back(normalized_to_predicted);

                lower_bounds[normalized_to_predicted] = 0;
                auto const arc_cost = (normalized_markers[normalized_idx] - prediction[predicted_idx]).norm();
                costs[normalized_to_predicted] = arc_cost;
            }
        }

        // note: create solver after graph, because it creates internal data structures based on it
        //       this produces a SIGSEGV in solver.lowerMap(lowerBounds), because the library does no
        //       bounds checks and does not even compare lowerMap size with internal data structure...
        // solver (the only one which supports float cost)
        lemon::CapacityScaling<decltype(graph), int, float> solver{graph};
        // set all arc capacities to 1
        solver.upperMap(lemon::constMap<decltype(graph)::ArcIt, int>(1));
        solver.lowerMap(lower_bounds);
        solver.costMap(costs);

        int n_of_markers = normalized_markers.size();

        // flow from source to sink = 19 (so 'one' flow per marker pair)
        solver.stSupply(source, sink, n_of_markers);

        auto const res = solver.run();
        switch (res) {
            case decltype(solver)::INFEASIBLE:
                throw std::runtime_error{"bad solver result: infeasible solution"};
            case decltype(solver)::OPTIMAL:
                break;
            case decltype(solver)::UNBOUNDED:
                throw std::runtime_error{"bad solver result: unbounded solution"};
            default:
                throw std::runtime_error{"unexpected solver.run() result: " + std::to_string(res)};
        }

        // find the marker pairs which have a non-zero flow between them...
        std::vector<marker_label> labels{};
        for (auto normalized_idx = 0ul; normalized_idx < normalized_nodes.size(); normalized_idx++) {
            bool has_label = false;

            for (auto predicted_idx = 0ul; predicted_idx < predicted_nodes.size(); predicted_idx++) {

                auto const arc = arcs[normalized_idx * n_of_markers + predicted_idx];
                auto const flow = solver.flow(arc);

                if (flow > 0) {
                    // due to the nature of the problem, the flow is 1 for at most one arc from the normalized side
                    // the push order is order of normalized nodes / markers
                    // the pushed marker label is based on predicted index (this specific order is specified by CNN)
                    labels.push_back(static_cast<marker_label>(predicted_idx));
                    has_label = true;
                }
            }

            // the current normalized marker has no arc to any of the predicted ones, so this one is not
            // part of the hand
            if (!has_label) {
                labels.push_back(marker_label::no_label);
            }
        }

        matching_result result;
        result.labels = std::move(labels);
        result.total_matching_cost = solver.totalCost<float>();
        return result;
    }
