#pragma once

#include "pipeline-tree-planner.h"

struct pipeline_tree_runtime_node {
    int parent;
    int depth;
    llama_token token;
    std::vector<float> hidden;
    std::vector<float> logits;
};

static int run_pipeline_tree(const options & opts, llama_context * early, tail_decoder & tail,
        const llama_vocab * vocab, int n_vocab, int n_embd, int n_prompt,
        llama_batch & early_batch, llama_batch & tail_batch, std::vector<float> early_logits,
        std::vector<float> target_logits, double prompt_early_ms, double prompt_tail_ms) {
    using timer = std::chrono::steady_clock;
    auto elapsed = [](timer::time_point start) {
        return std::chrono::duration<double, std::milli>(timer::now() - start).count();
    };
    pipeline_tree::model planner;
    if (!planner.load(opts.tree_cost_profile, opts.tree_coverage_profile, opts.tree_empirical_position)) {
        std::fprintf(stderr, "pipeline tree profile error: %s\n", planner.error.c_str());
        return 2;
    }

    auto ranked_token = [&](const std::vector<float> & logits, int rank) {
        std::vector<int> order(n_vocab);
        for (int i = 0; i < n_vocab; ++i) { order[i] = i; }
        std::partial_sort(order.begin(), order.begin() + rank + 1, order.end(), [&](int a, int b) {
            return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
        });
        return (llama_token) order[rank];
    };
    auto top1_probability = [&](const std::vector<float> & logits) {
        const llama_token top = argmax_token(logits.data(), n_vocab);
        const float maximum = logits[top];
        double total = 0.0;
        for (float value : logits) { total += std::exp(value - maximum); }
        return 1.0 / total;
    };

    int generated_count = 0, drafted = 0, accepted_total = 0, verifies = 0, bonus_count = 0;
    int leaves_total = 0, window_total = 0;
    double draft_ms = 0.0, verify_ms = 0.0, match_ms = 0.0, rollback_ms = 0.0, correction_ms = 0.0;
    double planning_ms = 0.0, prepare_ms = 0.0;
    bool stop = false, pending = false;
    std::vector<float> pending_hidden;
    std::vector<llama_token> generated;
    logits_dump diagnostic;
    if (!opts.logits_prefix.empty() && !diagnostic.open(opts.logits_prefix + ".pipeline", n_vocab)) { return 2; }
    tail.reset_generation_read_stats();
    const auto start = timer::now();
    std::fprintf(stderr, "layerskip: pipeline tree DP, max-nodes=%d, max-depth=%d, top-k=%d, draft-node-ms=%.3f\n",
            opts.tree_max_nodes, opts.draft_max, opts.tree_top_k, planner.draft_node_ms);
    if (opts.tree_confidence_width) {
        std::fprintf(stderr, "layerskip: confidence width, graphless Q2 reads before Draft, graph layout after Draft\n");
    }
    if (opts.tree_online_gain) {
        std::fprintf(stderr, "layerskip: online gain, expand only when predicted round throughput improves\n");
    }
    if (opts.tree_empirical_position) {
        std::fprintf(stderr, "layerskip: empirical position calibration, probability bin + depth + Target rank, prior=30\n");
    }
    if (opts.tree_probability_width) {
        std::fprintf(stderr, "layerskip: forced probability width, p>=0.7:1 p>=0.5:2 else:4, no gain stop, depth-limit=%d, validate at capacity\n", opts.draft_max);
    }

    while (generated_count < opts.n_predict && !stop) {
        const llama_pos base = n_prompt + generated_count;
        const int max_depth = opts.tree_probability_width ? opts.draft_max : std::min(opts.draft_max, opts.n_predict - generated_count);
        const auto planning_start = timer::now();
        const double root_p = top1_probability(early_logits);
        pipeline_tree::plan plan = opts.tree_confidence_width ? pipeline_tree::plan{} : opts.tree_io_balance ?
                planner.choose_io_balance(opts.tree_max_nodes, max_depth, opts.tree_top_k, pending, root_p, opts.tree_io_target_ratio) :
                opts.tree_adaptive_capacity ?
                planner.choose_adaptive(opts.tree_max_nodes, max_depth, opts.tree_top_k, pending, root_p) :
                planner.choose(opts.tree_max_nodes, max_depth, opts.tree_top_k, pending, root_p);
        planning_ms += elapsed(planning_start);
        if (!opts.tree_confidence_width && plan.nodes.empty()) {
            std::fprintf(stderr, "pipeline tree planner returned no nodes\n");
            return 2;
        }
        std::vector<pipeline_tree_runtime_node> nodes;
        nodes.reserve(opts.tree_confidence_width ? opts.tree_max_nodes : plan.nodes.size());

        const int row_offset = pending ? 1 : 0;
        auto prepare_layout = [&]() -> bool {
        const auto prepare_start = timer::now();
        tail_batch.n_tokens = 0;
        std::vector<float> zero_hidden(n_embd, 0.0f);
        std::vector<llama_seq_id> all_ids{0};
        for (int i = 0; i < (int) plan.nodes.size(); ++i) { all_ids.push_back(i + 1); }
        if (pending) {
            batch_add_embd_seq_list(tail_batch, zero_hidden.data(), n_embd, base - 1, all_ids, true);
        }
        for (int i = 0; i < (int) plan.nodes.size(); ++i) {
            std::vector<llama_seq_id> descendants;
            for (int j = i; j < (int) plan.nodes.size(); ++j) {
                int ancestor = j;
                while (ancestor >= 0 && ancestor != i) { ancestor = plan.nodes[ancestor].parent; }
                if (ancestor == i) { descendants.push_back(j + 1); }
            }
            batch_add_embd_seq_list(tail_batch, zero_hidden.data(), n_embd, base + plan.nodes[i].depth, descendants, true);
        }
        for (int i = 0; i < (int) plan.nodes.size(); ++i) {
            if (!tail.memory_seq_rm(i + 1, -1, -1)) { return false; }
            tail.memory_seq_cp(0, i + 1, -1, -1);
        }
        std::vector<int8_t> all_true_outputs(tail_batch.n_tokens, 1);
        const bool ready = opts.tree_confidence_width ? tail.bind_graphless_budget_layout(tail_batch) :
                tail.start_budget_round(tail_batch.n_tokens, tail_batch.pos[0], &all_true_outputs, &tail_batch);
        if (!ready) {
            std::fprintf(stderr, "pipeline tree budget round failed\n");
            return false;
        }
        prepare_ms += elapsed(prepare_start);
        return true;
        };
        if (opts.tree_confidence_width) {
            const auto prepare_start = timer::now();
            if (!tail.start_graphless_budget_round()) { return 2; }
            prepare_ms += elapsed(prepare_start);
        } else if (!prepare_layout()) { return 2; }

        const auto draft_start = timer::now();
        pipeline_tree::online_width_frontier frontier(planner, opts.tree_max_nodes, max_depth, opts.tree_top_k, root_p, opts.tree_probability_width);
        for (int i = 0;; ++i) {
            pipeline_tree::node shape;
            double path_mass = 0.0;
            if (opts.tree_confidence_width) {
                const bool selected = opts.tree_online_gain ?
                        frontier.next_gain(shape, path_mass, pending, elapsed(draft_start)) : frontier.next(shape, path_mass);
                if (opts.tree_online_gain) {
                    const auto & decision = frontier.last_gain_decision();
                    std::fprintf(stderr, "online gain round=%d nodes=%zu candidates=%d base-tps=%.6f trial-tps=%.6f marginal-tokens=%.6f marginal-wall-ms=%.3f decision=%s\n",
                            verifies + 1, frontier.current_plan().nodes.size(), decision.candidates,
                            1000.0 * decision.baseline_rate, 1000.0 * decision.trial_rate,
                            decision.marginal_tokens, decision.marginal_wall_ms, decision.stop_reason.c_str());
                }
                if (!selected) {
                    if (opts.tree_probability_width) {
                        std::fprintf(stderr, "forced width round=%d nodes=%zu decision=%s\n", verifies + 1,
                                frontier.current_plan().nodes.size(), frontier.last_gain_decision().stop_reason.c_str());
                        if (frontier.last_gain_decision().stop_reason == "capacity") {
                            std::fprintf(stderr, "forced width capacity reached; validate existing tree, some branches may be shorter than depth %d\n", max_depth);
                        }
                    }
                    break;
                }
            } else {
                if (i >= (int) plan.nodes.size()) { break; }
                shape = plan.nodes[i];
            }
            const std::vector<float> & parent_logits = shape.parent < 0 ? early_logits : nodes[shape.parent].logits;
            if (shape.rank < 0 || shape.rank >= opts.tree_top_k) { return 2; }
            const llama_token token = ranked_token(parent_logits, shape.rank);
            const llama_seq_id src = shape.parent < 0 ? 0 : shape.parent + 1;
            const llama_seq_id dst = i + 1;
            if (!llama_memory_seq_rm(llama_get_memory(early), dst, -1, -1)) { return 2; }
            llama_memory_seq_cp(llama_get_memory(early), src, dst, -1, -1);
            early_batch.n_tokens = 0;
            batch_add_token_seqs(early_batch, token, base + shape.depth, dst, -1, true);
            if (early_diag_call("pipeline.draft", [&]() { return llama_decode(early, early_batch); }) != 0) {
                std::fprintf(stderr, "pipeline tree early decode failed\n");
                return 2;
            }
            const float * hidden = llama_get_embeddings_layer_inp(early, opts.exit_layer);
            pipeline_tree_runtime_node node{shape.parent, shape.depth, token, {}, {}};
            if (!hidden || !early_diag_call("pipeline.logits", [&]() { return copy_logits(early, 0, n_vocab, node.logits); })) {
                std::fprintf(stderr, "pipeline tree early output missing\n");
                return 2;
            }
            node.hidden.assign(hidden, hidden + n_embd);
            nodes.push_back(std::move(node));
            if (opts.tree_confidence_width) {
                const double p1 = top1_probability(nodes.back().logits);
                if (!frontier.observe(i, p1)) { return 2; }
                std::fprintf(stderr, "confidence node round=%d index=%d parent=%d depth=%d rank=%d parent-p=%.6f path-mass=%.6f next-p=%.6f\n",
                        verifies + 1, i, shape.parent, shape.depth + 1, shape.rank + 1,
                        top1_probability(parent_logits), path_mass, p1);
            }
        }
        const double measured_draft = elapsed(draft_start);
        draft_ms += measured_draft;
        planner.observe_draft(measured_draft, (int) nodes.size());
        if (opts.tree_confidence_width) {
            plan = frontier.current_plan();
            if (plan.nodes.empty()) { return 2; }
            if (opts.tree_online_gain) {
                planner.evaluate_online(plan, pending, measured_draft);
            } else {
                planner.evaluate(plan, pending);
            }
            if (!prepare_layout()) { return 2; }
        }
        int actual_depth = 0;
        for (const auto & shape : plan.nodes) { actual_depth = std::max(actual_depth, shape.depth + 1); }
        window_total += actual_depth;

        if (pending) {
            std::copy(pending_hidden.begin(), pending_hidden.end(), tail_batch.embd);
        }
        for (int i = 0; i < (int) nodes.size(); ++i) {
            std::copy(nodes[i].hidden.begin(), nodes[i].hidden.end(), tail_batch.embd + (size_t) (i + row_offset) * n_embd);
        }
        const auto verify_start = timer::now();
        tail.resume_budget_reads();
        if (!tail.decode(tail_batch)) {
            std::fprintf(stderr, "pipeline tree target decode failed\n");
            return 2;
        }
        std::vector<std::vector<float>> verified(tail_batch.n_tokens);
        for (int row = 0; row < tail_batch.n_tokens; ++row) {
            if (!tail.copy_logits(row, verified[row])) { return 2; }
        }
        const double measured_verify = elapsed(verify_start);
        verify_ms += measured_verify;
        ++verifies;
        drafted += (int) nodes.size();
        leaves_total += plan.leaves;

        const auto match_start = timer::now();
        std::vector<float> current = pending ? verified[0] : target_logits;
        pending = false;
        int parent = -1, accepted = 0;
        bool leaf_end = false;
        while (generated_count < opts.n_predict) {
            const llama_token expected = argmax_token(current.data(), n_vocab);
            if (!opts.logits_prefix.empty() && !diagnostic.write(vocab, current.data(), n_vocab, generated_count, n_prompt + generated_count)) { return 2; }
            int child = -1;
            for (int i = 0; i < (int) nodes.size(); ++i) {
                if (nodes[i].parent == parent && nodes[i].token == expected) { child = i; break; }
            }
            if (child < 0) {
                leaf_end = parent >= 0;
                for (const auto & node : nodes) { if (node.parent == parent) { leaf_end = false; break; } }
                break;
            }
            print_token(vocab, expected);
            generated.push_back(expected);
            ++generated_count;
            ++accepted;
            parent = child;
            current = verified[child + row_offset];
            if (!opts.ignore_eos && llama_vocab_is_eog(vocab, expected)) { stop = true; break; }
        }
        accepted_total += accepted;
        target_logits = current;
        match_ms += elapsed(match_start);
        std::fprintf(stderr, "pipeline tree round=%d nodes=%zu leaves=%d accepted=%d predicted-ms=%.3f predicted-wait-ms=%.3f expected-accepted=%.3f predicted-read-ms=%.3f predicted-execute-ms=%.3f draft-ms=%.3f verify-ms=%.3f\n",
                verifies, nodes.size(), plan.leaves, accepted, plan.wall_ms, plan.wait_ms, plan.expected_tokens,
                plan.read_ms, plan.compute_ms, measured_draft, measured_verify);
        if (opts.tree_adaptive_capacity) {
            int depth = 0;
            for (const auto & node : plan.nodes) { depth = std::max(depth, node.depth + 1); }
            std::fprintf(stderr, "adaptive tree round=%d explored=%d capacity=%d selected=%zu depth=%d predicted-stream-execute-ms=%.3f predicted-io-execute-overlap-ms=%.3f extrapolated=%d\n",
                    verifies, plan.explored_nodes, opts.tree_max_nodes, nodes.size(), depth,
                    plan.stream_compute_ms, plan.io_execute_overlap_ms, plan.timing_prediction_uncalibrated);
        }
        if (opts.tree_io_balance) {
            std::fprintf(stderr, "io balance round=%d nodes=%zu target-ratio=%.3f predicted-min-pair-ratio=%.6f predicted-mean-pair-ratio=%.6f achieved=%d stop=%s\n",
                    verifies, nodes.size(), opts.tree_io_target_ratio, plan.io_balance_min_ratio,
                    plan.io_balance_mean_ratio, plan.io_balance_achieved, plan.io_balance_stop_reason.c_str());
        }
        if (stop || generated_count >= opts.n_predict) { break; }

        const auto rollback_start = timer::now();
        const llama_seq_id active = parent < 0 ? 0 : parent + 1;
        const llama_pos next_pos = base + accepted;
        if (!llama_memory_seq_rm(llama_get_memory(early), active, next_pos, -1) || !tail.memory_seq_rm(active, next_pos, -1)) { return 2; }
        if (active != 0) {
            llama_memory_seq_keep(llama_get_memory(early), active);
            llama_memory_seq_cp(llama_get_memory(early), active, 0, -1, -1);
            tail.memory_seq_keep(active);
            tail.memory_seq_cp(active, 0, -1, -1);
        }
        llama_memory_seq_keep(llama_get_memory(early), 0);
        tail.memory_seq_keep(0);
        rollback_ms += elapsed(rollback_start);

        const auto correction_start = timer::now();
        const llama_token fallback = argmax_token(target_logits.data(), n_vocab);
        if (leaf_end) { ++bonus_count; }
        print_token(vocab, fallback);
        generated.push_back(fallback);
        ++generated_count;
        if ((!opts.ignore_eos && llama_vocab_is_eog(vocab, fallback)) || generated_count >= opts.n_predict) {
            stop = true;
        } else {
            early_batch.n_tokens = 0;
            batch_add_token(early_batch, fallback, next_pos, true);
            if (early_diag_call("pipeline.correction", [&]() { return llama_decode(early, early_batch); }) != 0 ||
                    !early_diag_call("pipeline.correction_logits", [&]() { return copy_logits(early, -1, n_vocab, early_logits); })) { return 2; }
            const float * hidden = llama_get_embeddings_layer_inp(early, opts.exit_layer);
            if (!hidden) { return 2; }
            pending_hidden.assign(hidden, hidden + n_embd);
            pending = true;
        }
        correction_ms += elapsed(correction_start);
    }

    const double seconds = elapsed(start) / 1000.0;
    std::fprintf(stderr, "\nlayerskip generation %d tokens in %.3f seconds, speed: %.3f t/s\n", generated_count, seconds, generated_count / seconds);
    std::fprintf(stderr, "n_drafted = %d\nn_accept = %d\nn_verify = %d\nn_bonus = %d\n", drafted, accepted_total, verifies, bonus_count);
    std::fprintf(stderr, "accept = %.3f%%\nwindow accept = %.3f%%\n", drafted ? 100.0 * accepted_total / drafted : 0.0,
            window_total ? 100.0 * accepted_total / window_total : 0.0);
    std::fprintf(stderr, "tree actual window sum = %d\n", window_total);
    std::fprintf(stderr, "tree nodes = %d\ntree leaves per verify = %.3f\n", drafted, verifies ? (double) leaves_total / verifies : 0.0);
    print_token_trace(generated);
    std::fprintf(stderr, "phase prompt early = %.3f ms\nphase prompt tail = %.3f ms\nphase draft build = %.3f ms\nphase tail verify = %.3f ms\nphase correction early = %.3f ms\nphase match = %.3f ms\nphase KV rollback = %.3f ms\n",
            prompt_early_ms, prompt_tail_ms, draft_ms, verify_ms, correction_ms, match_ms, rollback_ms);
    std::fprintf(stderr, "phase tree planning = %.3f ms\nphase tree layout/initial prepare = %.3f ms\n", planning_ms, prepare_ms);
    std::fprintf(stderr, "tail HTP preload calls = %llu\ntail HTP preload failures = %llu\ntail HTP preload bytes = %llu\ntail HTP preload time = %.3f ms\n",
            (unsigned long long) tail.htp_preload_calls, (unsigned long long) tail.htp_preload_failures,
            (unsigned long long) tail.htp_preload_bytes, tail.htp_preload_time_us / 1000.0);
    std::fprintf(stderr, "tail stream cache hits = %llu\ntail stream cache bytes = %llu\n",
            (unsigned long long) tail.cache_hits, (unsigned long long) tail.cache_bytes);
    std::fprintf(stderr, "tail budget admitted headroom = %llu bytes\ntail budget read time = %.3f ms\ntail budget wait time = %.3f ms\ntail budget failures = %llu\ntail budget maximum queued = %llu\ntail budget peak in-flight layers = %llu\n",
            (unsigned long long) tail.budget_admitted_bytes, tail.budget_read_time_us / 1000.0, tail.budget_wait_time_us / 1000.0,
            (unsigned long long) tail.budget_failures, (unsigned long long) tail.budget_max_queued, (unsigned long long) tail.budget_peak_inflight);
    tail.print_budget_read_stats();
    std::fprintf(stderr, "allocation budget used = %llu bytes, peak = %llu bytes, failures = %llu\n",
            (unsigned long long) ggml_backend_allocation_budget_used(), (unsigned long long) ggml_backend_allocation_budget_peak(),
            (unsigned long long) ggml_backend_allocation_budget_failures());
    return 0;
}
