#pragma once

// Optional at source-tree creation time: the bridge is supplied by the HTP work.
#if __has_include("batch-draft-device.h")
#include "batch-draft-device.h"
#define BATCH_GAIN_HAS_HTP_BRIDGE 1
#else
#define BATCH_GAIN_HAS_HTP_BRIDGE 0
#endif

static double batch_gain_prior_ms(bool htp, int rows) {
    static const int sizes[] = {1, 4, 8, 16};
    // Planner146 saved full-action costs. 1..8 observed; 9..16 scaled
    // extrapolation, documented in initial-costs.json. Online EMA unchanged.
    #include "cpu-priors.h"
    static const double dsp[] = {114.489, 127.450, 139.594, 160.307};
    rows = std::max(1, std::min(16, rows));
    if (!htp) return planned_initial().enabled ? planned_initial().cpu[rows] : cpu[rows-1];
    const double * values = dsp;
    if (rows <= sizes[0]) return values[0];
    for (int i = 1; i != 4; ++i) if (rows <= sizes[i]) {
        const double t = double(rows - sizes[i - 1]) / double(sizes[i] - sizes[i - 1]);
        return values[i - 1] + t * (values[i] - values[i - 1]);
    }
    return values[3];
}

static int batch_gain_leaf_delta(int rows, int distinct_new_leaf_parents) {
    // The virtual root is not a leaf, hence it is deliberately absent here.
    return rows - distinct_new_leaf_parents;
}

static bool batch_gain_unit_tests() {
    if(planned_initial().enabled){
        for(int b=1;b<=16;++b)if(batch_gain_prior_ms(false,b)!=planned_initial().cpu[b])return false;
        return batch_gain_leaf_delta(4,0)==4&&batch_gain_leaf_delta(4,2)==2;
    }
    const double eps = 1e-9;
    return std::abs(batch_gain_prior_ms(false, 1) - 42.0935) < 1e-5 &&
            std::abs(batch_gain_prior_ms(false, 4) - 92.5765) < 1e-5 &&
            std::abs(batch_gain_prior_ms(false, 8) - 174.55) < 1e-5 &&
            std::abs(batch_gain_prior_ms(false, 16) - 337.754060) < 1e-5 &&
            std::abs(batch_gain_prior_ms(true, 1) - 114.489) < eps &&
            std::abs(batch_gain_prior_ms(true, 4) - 127.450) < eps &&
            std::abs(batch_gain_prior_ms(true, 8) - 139.594) < eps &&
            std::abs(batch_gain_prior_ms(true, 16) - 160.307) < eps &&
            std::abs(batch_gain_prior_ms(false, 2) - 52.976) < 1e-5 &&
            batch_gain_leaf_delta(4, 0) == 4 && batch_gain_leaf_delta(4, 1) == 3 &&
            batch_gain_leaf_delta(4, 2) == 2;
}

struct batch_gain_ema {
    double cpu_ratio[17] = {};
    double htp_ratio[17] = {};
    double estimate(bool htp, int rows, double transfer_prior) const {
        const double ratio = (htp ? htp_ratio[rows] : cpu_ratio[rows]);
        return batch_gain_prior_ms(htp, rows) * (ratio > 0 ? ratio : 1.0) + (htp ? transfer_prior : 0.0);
    }
    void observe(bool htp, int rows, double forward) {
        double * ratio = htp ? htp_ratio : cpu_ratio;
        const double value = forward / std::max(0.001, batch_gain_prior_ms(htp, rows));
        ratio[rows] = ratio[rows] > 0 ? .8 * ratio[rows] + .2 * value : value;
    }
};

static int run_batch_gain_tree(const options & opts, llama_context * early, tail_decoder & tail,
        const llama_vocab * vocab, int n_vocab, int n_embd, int n_prompt,
        llama_batch & early_batch, llama_batch & tail_batch, std::vector<float> early_logits,
        std::vector<float> target_logits, double prompt_early_ms, double prompt_tail_ms) {
    using timer = std::chrono::steady_clock;
    auto elapsed = [](timer::time_point start) { return std::chrono::duration<double, std::milli>(timer::now() - start).count(); };
    const char * mode = std::getenv("LLAMA_BATCH_GAIN");
    const bool hybrid = mode && !std::strcmp(mode, "hybrid");
    gain_tree_cost cost{{}, opts.tree_verify_base_ms, opts.tree_node_ms, opts.tree_leaf_ms};
    std::fprintf(stderr,"batch initial source=%s legacy-fallback-base-ms=%.6f node-ms=%.6f leaf-ms=%.6f online-update=unchanged\n",planned_initial().enabled?"planner-exported-curve":"fallback",cost.base_ms,cost.node_ms,cost.leaf_ms);
    batch_gain_ema forwards;
#if BATCH_GAIN_HAS_HTP_BRIDGE
    batch_draft_htp bridge;
    bool bridge_ok = false;
    if (hybrid) {
        // Deliberately before the generation timer: bridge setup is not generation work.
        bridge_ok = bridge.init(opts.model, early, opts.exit_layer, opts.n_ctx, opts.n_batch,
                opts.n_ubatch, opts.tree_max_nodes + 1, opts.n_threads);
        if (!bridge_ok) { std::fprintf(stderr, "batch gain hybrid bridge initialization failed\n"); return 2; }
    }
#else
    if (hybrid) { std::fprintf(stderr, "batch gain hybrid requested but batch-draft-device.h is unavailable\n"); return 2; }
#endif
    double reliability = .9, draft_ms = 0, verify_ms = 0, correction_ms = 0, match_ms = 0, rollback_ms = 0;
    int feedback = 0, generated_count = 0, drafted = 0, accepted_total = 0, verifies = 0, bonus_count = 0, leaves_total = 0, gain_stops = 0, htp_count = 0, cpu_count = 0, htp_nodes = 0, cpu_nodes = 0;
    double state_sync_ms = 0;
    std::string batch_hist;
    bool stop = false, pending = false;
    logits_dump diagnostic;
    if (!opts.logits_prefix.empty() && !diagnostic.open(opts.logits_prefix + ".gain", n_vocab)) return 2;
    std::vector<float> pending_hidden;
    std::vector<llama_token> generated;
    if (opts.tail_budget_prefetch) tail.reset_generation_read_stats();
    const auto start = timer::now();
    std::fprintf(stderr, "layerskip: batch gain tree mode=%s, max-nodes=%d max-depth=%d top-k=%d; measured whole-batch priors enabled\n", mode, opts.tree_max_nodes, opts.draft_max, opts.tree_top_k);

    while (generated_count < opts.n_predict && !stop) {
        const int max_depth = std::min(opts.draft_max, opts.n_predict - generated_count);
        const llama_pos base = n_prompt + generated_count;
        if (opts.tail_budget_prefetch && !tail.start_graphless_budget_round()) return 2;
        std::vector<gain_tree_node> nodes; nodes.reserve(opts.tree_max_nodes);
        std::vector<gain_tree_candidate> frontier;
        std::vector<llama_token> root_candidates;
        auto add_frontier = [&](int parent, const std::vector<float> & logits) {
            std::vector<int> indices(n_vocab); for (int i = 0; i < n_vocab; ++i) indices[i] = i;
            std::partial_sort(indices.begin(), indices.begin() + opts.tree_top_k, indices.end(), [&](int a, int b) { return logits[a] == logits[b] ? a < b : logits[a] > logits[b]; });
            const double maximum = logits[indices[0]]; double mass = 0;
            for (int i = 0; i < opts.tree_top_k; ++i) mass += std::exp(logits[indices[i]] - maximum);
            auto & candidates = parent < 0 ? root_candidates : nodes[parent].candidates;
            for (int i = 0; i < opts.tree_top_k; ++i) {
                candidates.push_back(indices[i]);
                frontier.push_back({parent, indices[i], reliability * std::exp(logits[indices[i]] - maximum) / mass * (parent < 0 ? 1.0 : nodes[parent].probability)});
            }
        };
        auto phase = timer::now(); add_frontier(-1, early_logits);
        int leaves = 0; double gain = 1.0; int round_actions = 0, round_depth = 0;
        std::string round_hist, stop_reason = "frontier-exhausted";
        while (!frontier.empty() && (int)nodes.size() < opts.tree_max_nodes) {
            // A stable order makes every prefix a ready, independent batch: no selected row
            // depends on the output produced by another selected row.
            std::stable_sort(frontier.begin(), frontier.end(), [](const gain_tree_candidate & a, const gain_tree_candidate & b) { return a.probability > b.probability; });
            const int capacity = std::min({16, (int)frontier.size(), opts.tree_max_nodes - (int)nodes.size()});
            const int rows = (int)nodes.size() + (pending ? 1 : 0);
            int choose = 0; bool choose_htp = false; double best_rate = -1, chosen_delta = 0, chosen_gain = 0; int chosen_leaves = leaves;
            for (int b = 1; b <= capacity; ++b) {
                double sum = 0; std::vector<int> unique_parents;
                for (int i = 0; i < b; ++i) { sum += frontier[i].probability; if (frontier[i].parent >= 0 && std::find(unique_parents.begin(), unique_parents.end(), frontier[i].parent) == unique_parents.end() && nodes[frontier[i].parent].children.empty()) unique_parents.push_back(frontier[i].parent); }
                // Each output row is a leaf, except that expanding a distinct current
                // leaf consumes that parent leaf.  The virtual root is not a leaf.
                const int next_leaves = leaves + batch_gain_leaf_delta(b, (int)unique_parents.size());
                for (int device = 0; device < (hybrid ? 2 : 1); ++device) {
                    const bool htp = device != 0;
                    if (htp && b < 8) continue;
                    double transfer_prior = 0;
#if BATCH_GAIN_HAS_HTP_BRIDGE
                    if (htp) transfer_prior = bridge.transfer_prior_ms;
#endif
                    const double delta = forwards.estimate(htp, b, transfer_prior) + std::max(0.0, cost.estimate(rows + b, next_leaves) - cost.estimate(rows, leaves));
                    const double rate = sum / std::max(.001, delta);
                    if (rate > best_rate) { best_rate = rate; choose = b; choose_htp = htp; chosen_delta = delta; chosen_gain = sum; chosen_leaves = next_leaves; }
                }
            }
            const double current_rate = gain / std::max(.001, elapsed(phase) + cost.estimate(rows, leaves));
            if (!nodes.empty() && best_rate <= current_rate) { ++gain_stops; stop_reason = "marginal-rate<=current-gain-rate"; break; }
            if (!choose) return 2; // at least one action is always selected for an empty tree.
            std::vector<gain_tree_candidate> selected(frontier.begin(), frontier.begin() + choose);
            frontier.erase(frontier.begin(), frontier.begin() + choose);
            const auto action_start = timer::now();
            early_batch.n_tokens = 0;
            const int first = (int)nodes.size();
            for (int i = 0; i < choose; ++i) {
                const int depth = selected[i].parent < 0 ? 0 : nodes[selected[i].parent].depth + 1;
                const llama_seq_id seq = first + i + 1;
                if (!llama_memory_seq_rm(llama_get_memory(early), seq, -1, -1)) return 2;
                llama_memory_seq_cp(llama_get_memory(early), selected[i].parent < 0 ? 0 : selected[i].parent + 1, seq, -1, -1);
                batch_add_token_seqs(early_batch, selected[i].token, base + depth, seq, -1, true);
            }
            if (early_batch.n_tokens != choose) return 2;
            double forward_ms = 0, transfer_ms = 0;
            std::vector<std::vector<float>> htp_logits, htp_hidden;
            const auto forward_start = timer::now();
            if (choose_htp) {
#if BATCH_GAIN_HAS_HTP_BRIDGE
                if (!bridge.decode(early, early_batch, n_vocab, n_embd, htp_logits, htp_hidden, transfer_ms, forward_ms)) return 2;
#else
                return 2;
#endif
            } else if (llama_decode(early, early_batch) != 0) { std::fprintf(stderr, "batch gain CPU decode failed\n"); return 2; }
            // Copy every row before appending nodes; batch output pointers are transient.
            const float * hidden = choose_htp ? nullptr : llama_get_embeddings_layer_inp(early, opts.exit_layer);
            if (!choose_htp && !hidden) return 2;
            for (int i = 0; i < choose; ++i) {
                const int depth = selected[i].parent < 0 ? 0 : nodes[selected[i].parent].depth + 1;
                gain_tree_node node{selected[i].parent, depth, selected[i].token, selected[i].probability, {}, {}, {}, {}};
                if (choose_htp) {
                    if ((int)htp_logits.size() != choose || (int)htp_hidden.size() != choose || (int)htp_logits[i].size() != n_vocab || (int)htp_hidden[i].size() != n_embd) return 2;
                    node.logits = std::move(htp_logits[i]); node.hidden = std::move(htp_hidden[i]);
                } else {
                    if (!copy_logits(early, i, n_vocab, node.logits)) return 2;
                    node.hidden.assign(hidden + (size_t)i * n_embd, hidden + (size_t)(i + 1) * n_embd);
                }
                nodes.push_back(std::move(node));
            }
            // CPU's public timing includes the synchronized hidden/logit copies and
            // node materialization.  add_frontier remains part of action_ms below.
            if (!choose_htp) forward_ms = elapsed(forward_start);
            for (int i = 0; i < choose; ++i) if (selected[i].parent >= 0) nodes[selected[i].parent].children.push_back(first + i);
            for (int i = 0; i < choose; ++i) { const int index = first + i; round_depth = std::max(round_depth, nodes[index].depth + 1); if (nodes[index].depth + 1 < max_depth && (opts.ignore_eos || !llama_vocab_is_eog(vocab, nodes[index].token))) add_frontier(index, nodes[index].logits); }
            const double action_ms = elapsed(action_start);
            // CPU's measured whole action deliberately includes KV cloning, batch setup,
            // output copies and node materialization.  HTP keeps forward and handoff
            // measurements separate because transfer is accounted as its own term.
            forwards.observe(choose_htp, choose, choose_htp ? std::max(.001, action_ms - transfer_ms) : action_ms);
            if (choose_htp) {
                ++htp_count; htp_nodes += choose; state_sync_ms += transfer_ms;
#if BATCH_GAIN_HAS_HTP_BRIDGE
                // bridge.decode() already updates its real-transfer EMA once.
#endif
            } else { ++cpu_count; cpu_nodes += choose; }
            leaves = chosen_leaves; gain += chosen_gain; ++round_actions;
            if (!round_hist.empty()) round_hist += ",";
            round_hist += (choose_htp ? "h" : "c") + std::to_string(choose);
            if (!batch_hist.empty()) batch_hist += ",";
            batch_hist += (choose_htp ? "h" : "c") + std::to_string(choose);
            std::fprintf(stderr, "batch gain action=%d b=%d device=%s gain=%.6f marginal-rate=%.8f measured-forward-ms=%.3f transfer-ms=%.3f total-ms=%.3f predicted-delta-ms=%.3f\n", round_actions, choose, choose_htp ? "htp" : "cpu", chosen_gain, best_rate, forward_ms, transfer_ms, action_ms, chosen_delta);
        }
        if (!frontier.empty() && (int)nodes.size() >= opts.tree_max_nodes) stop_reason = "capacity";
        draft_ms += elapsed(phase); phase = timer::now();
        for (int i = 0; i < (int)nodes.size(); ++i) { if (!tail.memory_seq_rm(i + 1, -1, -1)) return 2; tail.memory_seq_cp(0, i + 1, -1, -1); }
        tail_batch.n_tokens = 0;
        if (pending) { std::vector<llama_seq_id> ids{0}; for (int i = 0; i < (int)nodes.size(); ++i) ids.push_back(i + 1); batch_add_embd_seq_list(tail_batch, pending_hidden.data(), n_embd, base - 1, ids, true); }
        const int row_offset = pending ? 1 : 0;
        for (int i = 0; i < (int)nodes.size(); ++i) { std::vector<llama_seq_id> descendants; for (int j = i; j < (int)nodes.size(); ++j) { int ancestor = j; while (ancestor >= 0 && ancestor != i) ancestor = nodes[ancestor].parent; if (ancestor == i) descendants.push_back(j + 1); } batch_add_embd_seq_list(tail_batch, nodes[i].hidden.data(), n_embd, base + nodes[i].depth, descendants, true); }
        if (tail_batch.n_tokens != (int)nodes.size() + row_offset) return 2;
        if (opts.tail_budget_prefetch && !tail.bind_graphless_budget_layout(tail_batch)) return 2;
        if (!tail.decode(tail_batch)) return 2;
        std::vector<std::vector<float>> verified(tail_batch.n_tokens); for (int row = 0; row < tail_batch.n_tokens; ++row) if (!tail.copy_logits(row, verified[row])) return 2;
        const double tail_ms = elapsed(phase); verify_ms += tail_ms;
        std::fprintf(stderr,"tree verify cost feedback rows=%d leaves=%d predicted-ms=%.6f actual-ms=%.6f prior-samples=%zu\n",tail_batch.n_tokens,leaves,cost.estimate(tail_batch.n_tokens,leaves),tail_ms,cost.samples.size());
        cost.observe(tail_batch.n_tokens, leaves, tail_ms); ++verifies; drafted += (int)nodes.size(); leaves_total += leaves;
        phase = timer::now(); std::vector<float> current = pending ? verified[0] : target_logits; pending = false; int parent = -1, accepted = 0; bool leaf_end = false;
        while (generated_count < opts.n_predict) { const llama_token expected = argmax_token(current.data(), n_vocab); if (!opts.logits_prefix.empty() && !diagnostic.write(vocab, current.data(), n_vocab, generated_count, n_prompt + generated_count)) return 2; const auto & candidates = parent < 0 ? root_candidates : nodes[parent].candidates; if (!candidates.empty()) { const bool hit = std::find(candidates.begin(), candidates.end(), expected) != candidates.end(); reliability = .98 * reliability + .02 * (hit ? 1.0 : 0.0); ++feedback; } int child = -1; for (int i = 0; i < (int)nodes.size(); ++i) if (nodes[i].parent == parent && nodes[i].token == expected) { child = i; break; } if (child < 0) { leaf_end = parent >= 0 && nodes[parent].children.empty(); break; } print_token(vocab, expected); generated.push_back(expected); ++generated_count; ++accepted; parent = child; current = verified[child + row_offset]; if (!opts.ignore_eos && llama_vocab_is_eog(vocab, expected)) { stop = true; break; } }
        accepted_total += accepted; target_logits = current; match_ms += elapsed(phase);
        std::fprintf(stderr, "batch gain round=%d actions=%d nodes=%zu leaves=%d depth=%d batchhist=%s stop-reason=%s accepted=%d gain=%.3f verify-ms=%.3f reliability=%.3f htp-actions=%d\n", verifies, round_actions, nodes.size(), leaves, round_depth, round_hist.c_str(), stop_reason.c_str(), accepted, gain, tail_ms, reliability, htp_count);
        if (stop || generated_count >= opts.n_predict) break;
        phase = timer::now(); const llama_seq_id active = parent < 0 ? 0 : parent + 1; const llama_pos next_pos = base + accepted;
        if (!llama_memory_seq_rm(llama_get_memory(early), active, next_pos, -1) || !tail.memory_seq_rm(active, next_pos, -1)) return 2;
        if (active != 0) { llama_memory_seq_keep(llama_get_memory(early), active); llama_memory_seq_cp(llama_get_memory(early), active, 0, -1, -1); tail.memory_seq_keep(active); tail.memory_seq_cp(active, 0, -1, -1); }
        llama_memory_seq_keep(llama_get_memory(early), 0); tail.memory_seq_keep(0); rollback_ms += elapsed(phase); phase = timer::now();
        const llama_token fallback = argmax_token(target_logits.data(), n_vocab); if (leaf_end) ++bonus_count; print_token(vocab, fallback); generated.push_back(fallback); ++generated_count;
        if ((!opts.ignore_eos && llama_vocab_is_eog(vocab, fallback)) || generated_count >= opts.n_predict) stop = true;
        else { early_batch.n_tokens = 0; batch_add_token(early_batch, fallback, next_pos, true); if (llama_decode(early, early_batch) != 0 || !copy_logits(early, -1, n_vocab, early_logits)) return 2; const float * hidden = llama_get_embeddings_layer_inp(early, opts.exit_layer); if (!hidden) return 2; pending_hidden.assign(hidden, hidden + n_embd); pending = true; }
        correction_ms += elapsed(phase);
    }
    const double seconds = elapsed(start) / 1000.; if (opts.tail_budget_prefetch) tail.print_budget_read_stats();
    std::fprintf(stderr, "\nlayerskip generation %d tokens in %.3f seconds, speed: %.3f t/s\n", generated_count, seconds, generated_count / seconds);
    std::fprintf(stderr, "n_drafted = %d\nn_accept = %d\nn_verify = %d\nn_bonus = %d\n", drafted, accepted_total, verifies, bonus_count);
    std::fprintf(stderr, "accept = %.3f%%\nwindow accept = %.3f%%\n", drafted ? 100.0 * accepted_total / drafted : 0., verifies ? 100.0 * accepted_total / (verifies * opts.draft_max) : 0.);
    std::fprintf(stderr, "tree nodes = %d\ntree leaves per verify = %.3f\ntree gain stops = %d\ntree calibration observations = %d\n", drafted, verifies ? double(leaves_total) / verifies : 0., gain_stops, feedback);
    std::fprintf(stderr, "batch-gain summary cpu-actions=%d htp-actions=%d cpu-nodes=%d htp-nodes=%d state-sync-ms=%.3f batchhist=%s\n", cpu_count, htp_count, cpu_nodes, htp_nodes, state_sync_ms, batch_hist.c_str());
    print_token_trace(generated);
    std::fprintf(stderr, "phase prompt early = %.3f ms\nphase prompt tail = %.3f ms\nphase draft build = %.3f ms\nphase tail verify = %.3f ms\nphase correction early = %.3f ms\nphase match = %.3f ms\nphase KV rollback = %.3f ms\n", prompt_early_ms, prompt_tail_ms, draft_ms, verify_ms, correction_ms, match_ms, rollback_ms);
    std::fprintf(stderr, "tail HTP preload calls = %llu\ntail HTP preload failures = %llu\ntail HTP preload bytes = %llu\ntail HTP preload time = %.3f ms\n", (unsigned long long) tail.htp_preload_calls, (unsigned long long) tail.htp_preload_failures, (unsigned long long) tail.htp_preload_bytes, tail.htp_preload_time_us / 1000.0);
    return 0;
}
