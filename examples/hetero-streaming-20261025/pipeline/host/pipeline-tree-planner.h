#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace pipeline_tree {

struct node {
    int parent;
    int depth;
    int rank;
};

struct plan {
    std::vector<node> nodes;
    double expected_tokens = 0.0;
    double wall_ms = 0.0;
    double wait_ms = 0.0;
    double draft_ms = 0.0;
    double read_ms = 0.0;
    double compute_ms = 0.0;
    double stream_compute_ms = 0.0;
    double io_execute_overlap_ms = 0.0;
    bool timing_prediction_uncalibrated = false;
    int explored_nodes = 0;
    bool io_balance_achieved = false;
    double io_balance_min_ratio = 0.0;
    double io_balance_mean_ratio = 0.0;
    std::string io_balance_stop_reason;
    int leaves = 0;
};

class model {
public:
    std::string error;
    double draft_node_ms = 1.0;

    bool load(const std::string & cost_file, const std::string & coverage_file, bool position_calibration = false) {
        error.clear();
        costs_.clear();
        global_.clear();
        bins_.assign(10, std::vector<double>());
        bin_total_.assign(10, 0.0);
        position_bins_.clear();
        position_total_.clear();
        position_calibration_ = position_calibration;
        std::ifstream costs(cost_file);
        if (!costs) { return fail("cannot open cost file"); }
        std::string line;
        std::getline(costs, line);
        std::vector<double> draft_per_node;
        while (std::getline(costs, line)) {
            std::istringstream in(line);
            std::string kind, event;
            int layer = 0, tokens = 0, samples = 0;
            double median = 0.0;
            if (!(in >> kind >> event >> layer >> tokens >> samples >> median) || tokens <= 0 || !std::isfinite(median)) { continue; }
            if (event == "build") {
                draft_per_node.push_back(median / tokens);
            } else if (event == "queue_read" || event == "queue_prepare" || event == "queue_release" || event == "execute_sync") {
                costs_[event][layer][tokens].push_back(median);
            }
        }
        if (draft_per_node.empty()) { return fail("cost file has no draft build rows"); }
        std::sort(draft_per_node.begin(), draft_per_node.end());
        draft_node_ms = draft_per_node[draft_per_node.size() / 2];

        std::ifstream coverage(coverage_file);
        if (!coverage) { return fail("cannot open coverage file"); }
        int observations = 0;
        while (std::getline(coverage, line)) {
            const double p = field_double(line, "p=", -1.0);
            const int rank = field_int(line, "target_rank=", 0);
            if (!accepted_or_first_reject(line)) { continue; }
            if (!std::isfinite(p) || p < 0.0 || p > 1.0 || rank <= 0) { continue; }
            const int bin = std::min(9, static_cast<int>(p * 10.0));
            if (static_cast<int>(global_.size()) < rank) { global_.resize(rank, 0.0); }
            if (static_cast<int>(bins_[bin].size()) < rank) { bins_[bin].resize(rank, 0.0); }
            global_[rank - 1] += 1.0;
            bins_[bin][rank - 1] += 1.0;
            bin_total_[bin] += 1.0;
            const int position = field_int(line, "position=", 0);
            if (position >= 1 && position <= 10) {
                const auto key = std::make_pair(position, bin);
                auto & counts = position_bins_[key];
                if (static_cast<int>(counts.size()) < rank) { counts.resize(rank, 0.0); }
                counts[rank - 1] += 1.0;
                position_total_[key] += 1.0;
            }
            ++observations;
        }
        if (observations == 0) { return fail("coverage file has no target ranks"); }
        double total = 0.0;
        for (double v : global_) { total += v; }
        for (double & v : global_) { v /= total; }

        for (int layer = 8; layer <= 24; ++layer) {
            if (!has_cost("queue_read", layer) || !has_cost("execute_sync", layer)) {
                return fail("cost file misses streamed Q2 profile layer " + std::to_string(layer));
            }
        }
        for (int layer = 25; layer <= 31; ++layer) {
            if (!has_cost("execute_sync", layer)) {
                return fail("cost file misses resident profile layer " + std::to_string(layer));
            }
        }
        return true;
    }

    void observe_draft(double ms, int nodes) {
        if (ms > 0.0 && nodes > 0) { draft_node_ms = 0.8 * draft_node_ms + 0.2 * ms / nodes; }
    }

    void evaluate(plan & value, bool pending) const {
        finish_cost(value, pending);
    }

    // draft_ms is the measured cumulative Draft time when it is available.
    void evaluate_online(plan & value, bool pending, double draft_ms = -1.0) const {
        finish_cost(value, pending, draft_ms);
    }

    double global_rank_mass(int rank) const {
        return rank >= 0 && rank < static_cast<int>(global_.size()) ? global_[rank] : 0.0;
    }

    double root_rank_mass(double p, int rank) const {
        if (rank < 0 || global_.empty()) { return 0.0; }
        const int bin = std::min(9, std::max(0, static_cast<int>(p * 10.0)));
        const double local = rank < static_cast<int>(bins_[bin].size()) ? bins_[bin][rank] : 0.0;
        const double prior = rank < static_cast<int>(global_.size()) ? global_[rank] : 0.0;
        return (local + 10.0 * prior) / (bin_total_[bin] + 10.0);
    }

    double candidate_rank_mass(double p, int rank, int position) const {
        const double prior = root_rank_mass(p, rank);
        if (!position_calibration_ || rank < 0 || position < 1 || !std::isfinite(p)) { return prior; }
        const int bin = std::min(9, std::max(0, static_cast<int>(p * 10.0)));
        const auto key = std::make_pair(position, bin);
        const auto found = position_bins_.find(key);
        if (found == position_bins_.end()) { return prior; }
        const double count = rank < static_cast<int>(found->second.size()) ? found->second[rank] : 0.0;
        return (count + 30.0 * prior) / (position_total_.at(key) + 30.0);
    }

    // Values above the measured token range are conservative predictions.
    double predicted_cost_ms(const std::string & event, int layer, int tokens) const {
        return has_cost(event, layer) && tokens > 0 ? cost(event, layer, tokens) : 0.0;
    }

    plan choose(int max_nodes, int max_depth, int top_k, bool pending, double root_top1_probability) const {
        plan out;
        if (!error.empty() || max_nodes < 1 || max_depth < 1 || top_k < 1 || global_.empty()) { return out; }
        const int ranks = std::min(top_k, static_cast<int>(global_.size()));
        if (ranks == 0) { return out; }
        const int nmax = max_nodes;
        // value[d] is a forest of nodes at depth d; its budget includes those nodes.
        std::vector<std::vector<double>> value(max_depth + 1, std::vector<double>(nmax + 1, 0.0));
        std::vector<std::vector<choice>> pick(max_depth, std::vector<choice>(nmax + 1));
        for (int d = max_depth - 1; d >= 0; --d) {
            for (int budget = 1; budget <= nmax; ++budget) {
                double best = 0.0;
                int used_best = 0;
                std::vector<int> best_allocations(ranks, 0);
                std::vector<double> dp(budget + 1, -1.0e300), next(budget + 1, -1.0e300);
                std::vector<std::vector<int>> back(ranks, std::vector<int>(budget + 1, -1));
                dp[0] = 0.0;
                for (int r = 0; r < ranks; ++r) {
                    for (int used = 0; used <= budget; ++used) {
                        if (dp[used] < -1.0e200) { continue; }
                        for (int child_budget = 1; used + child_budget <= budget; ++child_budget) {
                            const double child = 1.0 + value[d + 1][child_budget - 1];
                            const double candidate = dp[used] + global_rank_mass(r) * child;
                            if (candidate > next[used + child_budget]) {
                                next[used + child_budget] = candidate;
                                back[r][used + child_budget] = child_budget;
                            }
                        }
                    }
                    for (int used = 0; used <= budget; ++used) {
                        if (next[used] > best) {
                            best = next[used]; used_best = used;
                            best_allocations = rebuild_allocations(back, r + 1, used);
                            best_allocations.resize(ranks, 0);
                        }
                    }
                    dp.swap(next);
                    std::fill(next.begin(), next.end(), -1.0e300);
                }
                value[d][budget] = best;
                pick[d][budget] = {used_best, best_allocations};
            }
        }

        double best_rate = -1.0;
        for (int budget = 1; budget <= nmax; ++budget) {
            double accepted = 0.0;
            std::vector<int> roots(ranks, 0);
            std::vector<double> dp(budget + 1, -1.0e300), next(budget + 1, -1.0e300);
            std::vector<std::vector<int>> back(ranks, std::vector<int>(budget + 1, -1));
            dp[0] = 0.0;
            for (int r = 0; r < ranks; ++r) {
                for (int at = 0; at <= budget; ++at) {
                    if (dp[at] < -1.0e200) { continue; }
                    for (int child_budget = 1; at + child_budget <= budget; ++child_budget) {
                        const double candidate = dp[at] + root_rank_mass(root_top1_probability, r) * (1.0 + value[1][child_budget - 1]);
                        if (candidate > next[at + child_budget]) {
                            next[at + child_budget] = candidate;
                            back[r][at + child_budget] = child_budget;
                        }
                    }
                }
                for (int at = 0; at <= budget; ++at) {
                    if (next[at] > accepted) {
                        accepted = next[at];
                        roots = rebuild_allocations(back, r + 1, at);
                        roots.resize(ranks, 0);
                    }
                }
                dp.swap(next);
                std::fill(next.begin(), next.end(), -1.0e300);
            }
            plan candidate;
            candidate.expected_tokens = accepted;
            for (int r = 0; r < ranks; ++r) { if (roots[r] > 0) { append_node(candidate.nodes, -1, 0, r, roots[r] - 1, pick, max_depth); } }
            finish_cost(candidate, pending);
            const double correction_ms = 0.25 * draft_node_ms;
            const double rate = (1.0 + candidate.expected_tokens) / std::max(0.001, candidate.wall_ms + correction_ms);
            if (rate > best_rate) { best_rate = rate; out = candidate; }
        }
        return out;
    }

    // max_nodes is a safety limit. The planner may stop before it.
    plan choose_adaptive(int max_nodes, int max_depth, int top_k, bool pending, double root_top1_probability) const {
        plan best;
        if (!error.empty() || max_nodes < 1 || max_depth < 1 || top_k < 1 || global_.empty()) { return best; }
        const int ranks = std::min(top_k, static_cast<int>(global_.size()));
        if (ranks == 0) { return best; }

        struct candidate {
            int parent;
            int depth;
            int rank;
            double probability;
        };
        std::vector<candidate> frontier;
        frontier.push_back({-1, 0, 0, root_rank_mass(root_top1_probability, 0)});
        plan current;
        double best_rate = -1.0;
        int explored = 0;

        while (!frontier.empty() && static_cast<int>(current.nodes.size()) < max_nodes) {
            int picked = -1;
            double picked_score = -1.0;
            plan picked_plan;
            for (int i = 0; i < static_cast<int>(frontier.size()); ++i) {
                if (frontier[i].probability <= 0.0) { continue; }
                plan trial = current;
                trial.nodes.push_back({frontier[i].parent, frontier[i].depth, frontier[i].rank});
                trial.expected_tokens += frontier[i].probability;
                finish_cost(trial, pending);
                const double score = frontier[i].probability / std::max(0.001, trial.wall_ms);
                if (score > picked_score) {
                    picked = i;
                    picked_score = score;
                    picked_plan = std::move(trial);
                }
            }
            if (picked < 0) { break; }

            const double correction_ms = 0.25 * draft_node_ms;
            const double rate = (1.0 + picked_plan.expected_tokens) / std::max(0.001, picked_plan.wall_ms + correction_ms);
            const candidate added = frontier[picked];
            frontier.erase(frontier.begin() + picked);
            current = std::move(picked_plan);
            ++explored;
            if (rate > best_rate + 1.0e-12 ||
                    (std::fabs(rate - best_rate) <= 1.0e-12 && current.io_execute_overlap_ms > best.io_execute_overlap_ms)) {
                best_rate = rate;
                best = current;
            }

            // Add ranks in order so every selected sibling set remains a prefix.
            if (added.rank + 1 < ranks) {
                const double sibling_p = added.parent < 0 ?
                    root_rank_mass(root_top1_probability, added.rank + 1) :
                    node_probability(current, added.parent, root_top1_probability) * global_rank_mass(added.rank + 1);
                frontier.push_back({added.parent, added.depth, added.rank + 1, sibling_p});
            }
            if (added.depth + 1 < max_depth) {
                frontier.push_back({static_cast<int>(current.nodes.size()) - 1, added.depth + 1, 0,
                    added.probability * global_rank_mass(0)});
            }
        }
        best.explored_nodes = explored;
        return best;
    }

    // Balance streamed layer L execution against the read of layer L+1. Draft is not part of this test.
    plan choose_io_balance(int max_nodes, int max_depth, int top_k, bool pending, double root_top1_probability,
            double target_ratio = 1.0) const {
        plan out;
        if (!std::isfinite(target_ratio) || target_ratio <= 0.0) {
            out.io_balance_stop_reason = "invalid_target";
            return out;
        }
        if (!error.empty() || max_nodes < 1 || max_depth < 1 || top_k < 1 || global_.empty()) { return out; }
        const int ranks = std::min(top_k, static_cast<int>(global_.size()));
        if (ranks == 0) { return out; }

        struct candidate {
            int parent;
            int depth;
            int rank;
            double probability;
        };
        std::vector<candidate> frontier;
        frontier.push_back({-1, 0, 0, root_rank_mass(root_top1_probability, 0)});

        while (!frontier.empty() && static_cast<int>(out.nodes.size()) < max_nodes) {
            int picked = -1;
            for (int i = 0; i < static_cast<int>(frontier.size()); ++i) {
                if (frontier[i].probability > 0.0 && (picked < 0 || frontier[i].probability > frontier[picked].probability)) {
                    picked = i;
                }
            }
            if (picked < 0) {
                out.io_balance_stop_reason = "frontier_exhausted";
                break;
            }

            const candidate added = frontier[picked];
            frontier.erase(frontier.begin() + picked);
            out.nodes.push_back({added.parent, added.depth, added.rank});
            out.expected_tokens += added.probability;
            ++out.explored_nodes;
            if (added.rank + 1 < ranks) {
                const double sibling_p = added.parent < 0 ?
                    root_rank_mass(root_top1_probability, added.rank + 1) :
                    node_probability(out, added.parent, root_top1_probability) * global_rank_mass(added.rank + 1);
                frontier.push_back({added.parent, added.depth, added.rank + 1, sibling_p});
            }
            if (added.depth + 1 < max_depth) {
                frontier.push_back({static_cast<int>(out.nodes.size()) - 1, added.depth + 1, 0,
                    added.probability * global_rank_mass(0)});
            }

            io_balance(out, pending);
            if (out.io_balance_min_ratio >= target_ratio) {
                out.io_balance_achieved = true;
                out.io_balance_stop_reason = "target";
                break;
            }
        }
        if (out.io_balance_stop_reason.empty()) {
            out.io_balance_stop_reason = static_cast<int>(out.nodes.size()) >= max_nodes ? "capacity" : "frontier_exhausted";
        }
        const int explored = out.explored_nodes;
        const bool achieved = out.io_balance_achieved;
        const double min_ratio = out.io_balance_min_ratio;
        const double mean_ratio = out.io_balance_mean_ratio;
        const std::string stop_reason = out.io_balance_stop_reason;
        finish_cost(out, pending);
        out.explored_nodes = explored;
        out.io_balance_achieved = achieved;
        out.io_balance_min_ratio = min_ratio;
        out.io_balance_mean_ratio = mean_ratio;
        out.io_balance_stop_reason = stop_reason;
        return out;
    }

private:
    typedef std::map<int, std::vector<double>> token_rows;
    typedef std::map<int, token_rows> layer_rows;
    struct choice { int used = 0; std::vector<int> allocations; };
    std::map<std::string, layer_rows> costs_;
    std::vector<double> global_;
    std::vector<std::vector<double>> bins_;
    std::vector<double> bin_total_;
    bool position_calibration_ = false;
    std::map<std::pair<int, int>, std::vector<double>> position_bins_;
    std::map<std::pair<int, int>, double> position_total_;

    bool fail(const std::string & message) { error = message; return false; }
    bool has_cost(const std::string & event, int layer) const {
        const auto e = costs_.find(event);
        return e != costs_.end() && e->second.find(layer) != e->second.end() && !e->second.at(layer).empty();
    }
    static int field_int(const std::string & s, const char * key, int fallback) {
        const std::size_t p = s.find(key);
        if (p == std::string::npos) { return fallback; }
        std::istringstream in(s.substr(p + std::string(key).size())); int v = fallback; in >> v; return v;
    }
    static double field_double(const std::string & s, const char * key, double fallback) {
        const std::size_t p = s.find(key);
        if (p == std::string::npos) { return fallback; }
        std::istringstream in(s.substr(p + std::string(key).size())); double v = fallback; in >> v; return v;
    }
    static bool accepted_or_first_reject(const std::string & s) {
        const std::size_t p = s.find("status=");
        if (p == std::string::npos) { return false; }
        std::istringstream in(s.substr(p + 7));
        std::string status;
        in >> status;
        return status == "accepted" || status == "first_reject";
    }
    static std::vector<int> rebuild_allocations(const std::vector<std::vector<int>> & back, int ranks, int used) {
        std::vector<int> allocations(ranks, 0);
        for (int r = ranks - 1; r >= 0; --r) {
            const int child = used >= 0 ? back[r][used] : -1;
            if (child > 0) { allocations[r] = child; used -= child; }
        }
        return allocations;
    }
    static void append_node(std::vector<node> & nodes, int parent, int depth, int rank, int descendants,
            const std::vector<std::vector<choice>> & pick, int max_depth) {
        const int index = static_cast<int>(nodes.size());
        nodes.push_back({parent, depth, rank});
        if (descendants == 0 || depth + 1 >= max_depth) { return; }
        const choice & c = pick[depth + 1][descendants];
        for (int r = 0; r < static_cast<int>(c.allocations.size()); ++r) {
            if (c.allocations[r] > 0) { append_node(nodes, index, depth + 1, r, c.allocations[r] - 1, pick, max_depth); }
        }
    }
    double cost(const std::string & event, int layer, int tokens) const {
        const token_rows & rows = costs_.at(event).at(layer);
        const auto hi = rows.lower_bound(tokens);
        if (hi == rows.begin()) { return average(hi->second); }
        if (hi == rows.end()) {
            const auto last = std::prev(hi);
            const double last_cost = average(last->second);
            // A layer's weights do not grow with the tree width.
            if (event == "queue_read") { return last_cost; }
            if (rows.size() == 1) { return last_cost * tokens / last->first; }
            const auto previous = std::prev(last);
            const double observed_slope = (last_cost - average(previous->second)) / (last->first - previous->first);
            const double baseline_slope = last_cost / last->first;
            return last_cost + std::max(0.0, std::max(observed_slope, baseline_slope)) * (tokens - last->first);
        }
        if (hi->first == tokens) { return average(hi->second); }
        const auto lo = std::prev(hi);
        const double a = average(lo->second), b = average(hi->second);
        return a + (b - a) * (tokens - lo->first) / (hi->first - lo->first);
    }
    bool cost_is_extrapolated(const std::string & event, int layer, int tokens) const {
        const token_rows & rows = costs_.at(event).at(layer);
        return !rows.empty() && tokens > rows.rbegin()->first;
    }
    static double average(const std::vector<double> & xs) {
        double sum = 0.0; for (double x : xs) { sum += x; } return sum / xs.size();
    }
    double node_probability(const plan & tree, int index, double root_top1_probability) const {
        double probability = 1.0;
        while (index >= 0) {
            const node & n = tree.nodes[index];
            probability *= n.parent < 0 ? root_rank_mass(root_top1_probability, n.rank) : global_rank_mass(n.rank);
            index = n.parent;
        }
        return probability;
    }
    void io_balance(plan & out, bool pending) const {
        const int tokens = std::max(1, static_cast<int>(out.nodes.size()) + (pending ? 1 : 0));
        double sum = 0.0;
        // Use the slowest pair. A mean can hide one IO-bound streamed layer.
        out.io_balance_min_ratio = std::numeric_limits<double>::infinity();
        for (int layer = 8; layer <= 23; ++layer) {
            const double read = cost("queue_read", layer + 1, tokens);
            const double execute = cost("execute_sync", layer, tokens);
            const double ratio = read > 0.0 ? execute / read : 0.0;
            out.io_balance_min_ratio = std::min(out.io_balance_min_ratio, ratio);
            sum += ratio;
        }
        out.io_balance_mean_ratio = sum / 16.0;
    }
    void finish_cost(plan & out, bool pending, double draft_ms_override = -1.0) const {
        out.wall_ms = 0.0;
        out.wait_ms = 0.0;
        out.draft_ms = 0.0;
        out.read_ms = 0.0;
        out.compute_ms = 0.0;
        out.stream_compute_ms = 0.0;
        out.io_execute_overlap_ms = 0.0;
        out.timing_prediction_uncalibrated = false;
        out.explored_nodes = 0;
        out.io_balance_achieved = false;
        out.io_balance_min_ratio = 0.0;
        out.io_balance_mean_ratio = 0.0;
        out.io_balance_stop_reason.clear();
        out.leaves = 0;
        const int tokens = std::max(1, static_cast<int>(out.nodes.size()) + (pending ? 1 : 0));
        const bool has_draft_override = std::isfinite(draft_ms_override) && draft_ms_override >= 0.0;
        out.draft_ms = has_draft_override ? draft_ms_override :
            out.nodes.size() * draft_node_ms + (pending ? 0.25 * draft_node_ms : 0.0);
        struct interval { double begin; double end; };
        std::vector<interval> reads;
        std::vector<interval> stream_executes;
        double reader_end = 0.0, exec_end = 0.0;
        double ready[17] = {};
        auto prepare_read = [&](int layer) {
            const double prepare = has_cost("queue_prepare", layer) ? cost("queue_prepare", layer, tokens) : 0.0;
            const double read = cost("queue_read", layer, tokens);
            out.timing_prediction_uncalibrated = out.timing_prediction_uncalibrated || cost_is_extrapolated("queue_read", layer, tokens) ||
                (has_cost("queue_prepare", layer) && cost_is_extrapolated("queue_prepare", layer, tokens));
            exec_end += prepare;
            const double read_begin = std::max(reader_end, exec_end);
            reader_end = read_begin + read;
            ready[layer - 8] = reader_end;
            out.read_ms += read;
            reads.push_back({read_begin, reader_end});
        };
        // The main thread prepares Q2 before Draft and refills after each release.
        prepare_read(8);
        prepare_read(9);
        exec_end += out.draft_ms;
        for (int layer = 8; layer <= 24; ++layer) {
            const double execute = cost("execute_sync", layer, tokens);
            const double release = has_cost("queue_release", layer) ? cost("queue_release", layer, tokens) : 0.0;
            out.timing_prediction_uncalibrated = out.timing_prediction_uncalibrated || cost_is_extrapolated("execute_sync", layer, tokens) ||
                (has_cost("queue_release", layer) && cost_is_extrapolated("queue_release", layer, tokens));
            out.wait_ms += std::max(0.0, ready[layer - 8] - exec_end);
            const double execute_begin = std::max(exec_end, ready[layer - 8]);
            stream_executes.push_back({execute_begin, execute_begin + execute});
            exec_end = execute_begin + execute + release;
            if (layer + 2 <= 24) { prepare_read(layer + 2); }
            out.compute_ms += execute;
            out.stream_compute_ms += execute;
        }
        for (int layer = 25; layer <= 31; ++layer) {
            const double execute = cost("execute_sync", layer, tokens);
            out.timing_prediction_uncalibrated = out.timing_prediction_uncalibrated || cost_is_extrapolated("execute_sync", layer, tokens);
            exec_end = std::max(exec_end, out.draft_ms) + execute;
            out.compute_ms += execute;
        }
        // compute_ms sums execute_sync for streamed and resident layers.
        out.wall_ms = exec_end;
        for (const interval & read : reads) {
            for (const interval & execute : stream_executes) {
                out.io_execute_overlap_ms += std::max(0.0, std::min(read.end, execute.end) - std::max(read.begin, execute.begin));
            }
        }
        for (int i = 0; i < static_cast<int>(out.nodes.size()); ++i) {
            bool leaf = true;
            for (const node & n : out.nodes) { if (n.parent == i) { leaf = false; break; } }
            if (leaf) { ++out.leaves; }
        }
    }
};

class online_width_frontier {
public:
    struct gain_decision {
        double baseline_rate = 0.0;
        double trial_rate = 0.0;
        double marginal_tokens = 0.0;
        double marginal_wall_ms = 0.0;
        int candidates = 0;
        std::string stop_reason;
    };

    online_width_frontier(const model & calibration, int cap, int max_depth, int top_k, double root_p, bool probability_width = false) :
            calibration_(calibration), cap_(std::max(0, cap)), max_depth_(std::max(0, max_depth)), top_k_(std::max(0, top_k)),
            probability_width_(probability_width) {
        if (cap_ > 0 && max_depth_ > 0 && top_k_ > 0 && std::isfinite(root_p) && root_p >= 0.0 && root_p <= 1.0) {
            frontier_.push_back({-1, 0, 0, calibration_.candidate_rank_mass(root_p, 0, 1), root_p});
        }
    }

    bool next(node & out, double & mass) {
        return next_impl(out, mass, false, -1.0, false);
    }

    // Select only when one more node improves predicted throughput.
    bool next_gain(node & out, double & mass, bool pending, double measured_draft_ms) {
        return next_impl(out, mass, pending, measured_draft_ms, true);
    }

    const gain_decision & last_gain_decision() const { return last_decision_; }

    static int probability_width(double p) {
        return p >= 0.7 ? 1 : p >= 0.5 ? 2 : 4;
    }

private:
    bool next_impl(node & out, double & mass, bool pending, double measured_draft_ms, bool online) {
        last_decision_ = gain_decision();
        if (static_cast<int>(plan_.nodes.size()) >= cap_) {
            last_decision_.stop_reason = "capacity";
            return false;
        }
        if (frontier_.empty()) {
            last_decision_.stop_reason = "frontier_exhausted";
            return false;
        }
        const bool measured_draft_valid = std::isfinite(measured_draft_ms) && measured_draft_ms >= 0.0;
        plan baseline = plan_;
        if (online && !plan_.nodes.empty() && measured_draft_valid) {
            calibration_.evaluate_online(baseline, pending, measured_draft_ms);
            last_decision_.baseline_rate = throughput(baseline);
        }
        int best = -1;
        plan best_trial;
        for (int i = 0; i < static_cast<int>(frontier_.size()); ++i) {
            if (std::isfinite(frontier_[i].mass) && (frontier_[i].mass > 0.0 || (probability_width_ && frontier_[i].mass == 0.0)) &&
                    (!online || measured_draft_valid || plan_.nodes.empty())) {
                ++last_decision_.candidates;
                if (!online) {
                    if (best < 0 || frontier_[i].mass > frontier_[best].mass) { best = i; }
                    continue;
                }
                plan trial = plan_;
                trial.nodes.push_back({frontier_[i].parent, frontier_[i].depth, frontier_[i].rank});
                trial.expected_tokens += frontier_[i].mass;
                calibration_.evaluate_online(trial, pending, measured_draft_valid ?
                    measured_draft_ms + calibration_.draft_node_ms : -1.0);
                const double rate = throughput(trial);
                if (best < 0 || rate > last_decision_.trial_rate) {
                    best = i;
                    best_trial = std::move(trial);
                    last_decision_.trial_rate = rate;
                }
            }
        }
        if (best < 0) {
            last_decision_.stop_reason = online && !measured_draft_valid && !plan_.nodes.empty() ? "invalid_draft_ms" : "frontier_exhausted";
            return false;
        }
        if (online) {
            last_decision_.marginal_tokens = frontier_[best].mass;
            last_decision_.marginal_wall_ms = best_trial.wall_ms - baseline.wall_ms;
        }
        if (online && !plan_.nodes.empty() &&
                !(last_decision_.trial_rate > last_decision_.baseline_rate + 1.0e-12)) {
            last_decision_.stop_reason = "no_gain";
            return false;
        }
        const candidate selected = frontier_[best];
        frontier_.erase(frontier_.begin() + best);
        out = {selected.parent, selected.depth, selected.rank};
        const int index = static_cast<int>(plan_.nodes.size());
        plan_.nodes.push_back(out);
        path_mass_.push_back(selected.mass);
        plan_.expected_tokens += selected.mass;
        observed_.push_back(false);
        mass = selected.mass;
        const int width = probability_width_ ? std::min(top_k_, probability_width(selected.parent_p)) : top_k_;
        if (selected.rank + 1 < width) {
            const double parent_mass = selected.parent < 0 ? 1.0 : path_mass_[selected.parent];
            const double sibling_mass = parent_mass * calibration_.candidate_rank_mass(selected.parent_p, selected.rank + 1, selected.depth + 1);
            if (std::isfinite(sibling_mass) && (sibling_mass > 0.0 || (probability_width_ && sibling_mass == 0.0))) {
                frontier_.push_back({selected.parent, selected.depth, selected.rank + 1, sibling_mass, selected.parent_p});
            }
        }
        last_index_ = index;
        if (online) {
            last_decision_.stop_reason = index == 0 ? "initial" : "expand";
        }
        return true;
    }

    double throughput(const plan & value) const {
        return (1.0 + value.expected_tokens) / std::max(0.001, value.wall_ms + 0.25 * calibration_.draft_node_ms);
    }

public:

    bool observe(int index, double p1) {
        if (index < 0 || index >= static_cast<int>(plan_.nodes.size()) || observed_[index] ||
                !std::isfinite(p1) || p1 < 0.0 || p1 > 1.0) {
            return false;
        }
        observed_[index] = true;
        const node & parent = plan_.nodes[index];
        if (parent.depth + 1 >= max_depth_) { return true; }
        const double child_mass = path_mass_[index] * calibration_.candidate_rank_mass(p1, 0, parent.depth + 2);
        if (std::isfinite(child_mass) && (child_mass > 0.0 || (probability_width_ && child_mass == 0.0))) {
            frontier_.push_back({index, parent.depth + 1, 0, child_mass, p1});
        }
        return true;
    }

    const plan & current_plan() const { return plan_; }
    int last_index() const { return last_index_; }

private:
    struct candidate {
        int parent;
        int depth;
        int rank;
        double mass;
        double parent_p;
    };
    const model & calibration_;
    int cap_;
    int max_depth_;
    int top_k_;
    bool probability_width_;
    int last_index_ = -1;
    plan plan_;
    std::vector<double> path_mass_;
    std::vector<bool> observed_;
    std::vector<candidate> frontier_;
    gain_decision last_decision_;
};

} // namespace pipeline_tree
