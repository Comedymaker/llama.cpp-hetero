#pragma once
#include "tile-cost.h"
#include "planned-initial.h"

struct gain_tree_candidate {
    int parent;
    llama_token token;
    double probability;
};

struct gain_tree_node {
    int parent;
    int depth;
    llama_token token;
    double probability;
    std::vector<float> hidden;
    std::vector<float> logits;
    std::vector<llama_token> candidates;
    std::vector<int> children;
};

struct gain_tree_cost {
    struct sample { int nodes; int leaves; double ms; };
    std::vector<sample> samples;
    double base_ms;
    double node_ms;
    double leaf_ms;
    double draft_ms = 100.0;
    int tile_rows = 0;
    double tile_ms = 0;

    double shape(int nodes,int leaves) const {
        if(planned_initial().enabled)return planned_initial().verification(nodes);
        return gain_shape_cost(nodes,leaves,node_ms,leaf_ms,tile_rows,tile_ms);
    }

    double estimate(int nodes, int leaves) const {
        if (samples.empty()) {
            if(planned_initial().enabled)return planned_initial().verification(nodes);
            return base_ms + shape(nodes,leaves);
        }
        const sample * nearest = &samples[0];
        int distance = std::abs(nearest->nodes - nodes) + std::abs(nearest->leaves - leaves);
        for (const auto & s : samples) {
            const int d = std::abs(s.nodes - nodes) + std::abs(s.leaves - leaves);
            if (d < distance) { nearest = &s; distance = d; }
        }
        return std::max(1.0, nearest->ms + shape(nodes,leaves)-shape(nearest->nodes,nearest->leaves));
    }

    void observe(int nodes, int leaves, double ms) {
        for (auto & s : samples) {
            if (s.nodes == nodes && s.leaves == leaves) {
                s.ms = 0.8 * s.ms + 0.2 * ms;
                return;
            }
        }
        samples.push_back({nodes, leaves, ms});
    }
};

#include "batch-gain-run.h"

static int run_gain_tree(const options & opts, llama_context * early, tail_decoder & tail,
        const llama_vocab * vocab, int n_vocab, int n_embd, int n_prompt,
        llama_batch & early_batch, llama_batch & tail_batch, std::vector<float> early_logits,
        std::vector<float> target_logits, double prompt_early_ms, double prompt_tail_ms) {
    const char * batch_gain_mode = std::getenv("LLAMA_BATCH_GAIN");
    if (batch_gain_mode && (!std::strcmp(batch_gain_mode, "cpu") || !std::strcmp(batch_gain_mode, "hybrid"))) {
        return run_batch_gain_tree(opts, early, tail, vocab, n_vocab, n_embd, n_prompt,
                early_batch, tail_batch, std::move(early_logits), std::move(target_logits),
                prompt_early_ms, prompt_tail_ms);
    }
    using timer = std::chrono::steady_clock;
    auto elapsed = [](timer::time_point start) {
        return std::chrono::duration<double, std::milli>(timer::now() - start).count();
    };
    gain_tree_cost cost{{}, opts.tree_verify_base_ms, opts.tree_node_ms, opts.tree_leaf_ms};
    const char *tile_setting=std::getenv("LLAMA_GAIN_TILE_ROWS");
    if(tile_setting){
        cost.tile_rows=std::atoi(tile_setting);
        const char *ms=std::getenv("LLAMA_GAIN_TILE_MS");cost.tile_ms=ms?std::atof(ms):0;
        if(cost.tile_rows!=32||!std::isfinite(cost.tile_ms)||cost.tile_ms<=0){std::fprintf(stderr,"invalid tile cost settings\n");return 2;}
    }
    double reliability = 0.9;
    int feedback = 0;
    int generated_count = 0, drafted = 0, accepted_total = 0, verifies = 0, bonus_count = 0;
    int leaves_total = 0, gain_stops = 0;
    double draft_ms = 0, verify_ms = 0, correction_ms = 0, match_ms = 0, rollback_ms = 0;
    bool stop = false, pending = false;
    logits_dump diagnostic;
    if (!opts.logits_prefix.empty() && !diagnostic.open(opts.logits_prefix + ".gain", n_vocab)) { return 2; }
    std::vector<float> pending_hidden;
    std::vector<llama_token> generated;
    if(opts.tail_budget_prefetch)tail.reset_generation_read_stats();
    const auto start = timer::now();
    std::fprintf(stderr,"gain tile model rows=%d tile-ms=%.3f residual-node-ms=%.3f leaf-ms=%.3f prior-not-measured-tree-curve=1\n",cost.tile_rows,cost.tile_ms,cost.node_ms,cost.leaf_ms);
    std::fprintf(stderr, "layerskip: gain-cost tree, max-nodes=%d, max-depth=%d, top-k=%d, prior-ms=%.3f/%.3f/%.3f\n",
            opts.tree_max_nodes, opts.draft_max, opts.tree_top_k, cost.base_ms, cost.node_ms, cost.leaf_ms);

    while (generated_count < opts.n_predict && !stop) {
        const int max_depth = std::min(opts.draft_max, opts.n_predict - generated_count);
        const llama_pos base = n_prompt + generated_count;
        if(opts.tail_budget_prefetch&&!tail.start_graphless_budget_round()){std::fprintf(stderr,"gain graphless prefetch failed\n");return 2;}
        std::vector<gain_tree_node> nodes;
        nodes.reserve(opts.tree_max_nodes);
        std::vector<gain_tree_candidate> frontier;
        std::vector<llama_token> root_candidates;
        auto add_frontier = [&](int parent, const std::vector<float> & logits) {
            std::vector<int> indices(n_vocab);
            for (int i = 0; i < n_vocab; ++i) { indices[i] = i; }
            std::partial_sort(indices.begin(), indices.begin() + opts.tree_top_k, indices.end(), [&](int a, int b) {
                return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
            });
            const double maximum = logits[indices[0]];
            double mass = 0;
            for (int i = 0; i < opts.tree_top_k; ++i) { mass += std::exp(logits[indices[i]] - maximum); }
            auto & candidates = parent < 0 ? root_candidates : nodes[parent].candidates;
            for (int i = 0; i < opts.tree_top_k; ++i) {
                const llama_token token = indices[i];
                candidates.push_back(token);
                const double conditional = reliability * std::exp(logits[token] - maximum) / mass;
                frontier.push_back({parent, token, conditional * (parent < 0 ? 1.0 : nodes[parent].probability)});
            }
        };
        auto phase = timer::now();
        add_frontier(-1, early_logits);
        int leaves = 0;
        double gain = 1.0;
        while (!frontier.empty() && (int) nodes.size() < opts.tree_max_nodes) {
            int best = -1;
            double best_rate = -1;
            int best_leaves = 0;
            const int rows = (int) nodes.size() + (pending ? 1 : 0);
            const double current_cost = elapsed(phase) + cost.estimate(rows, leaves);
            for (int i = 0; i < (int) frontier.size(); ++i) {
                const auto & c = frontier[i];
                const int next_leaves = leaves + (c.parent < 0 || !nodes[c.parent].children.empty() ? 1 : 0);
                const double delta = cost.draft_ms + std::max(0.0,
                        cost.estimate(rows + 1, next_leaves) - cost.estimate(rows, leaves));
                const double rate = c.probability / std::max(0.001, delta);
                if (rate > best_rate) { best = i; best_rate = rate; best_leaves = next_leaves; }
            }
            if (!nodes.empty() && best_rate <= gain / std::max(0.001, current_cost)) {
                ++gain_stops;
                break;
            }
            const auto candidate = frontier[best];
            frontier.erase(frontier.begin() + best);
            const int index = (int) nodes.size();
            const int depth = candidate.parent < 0 ? 0 : nodes[candidate.parent].depth + 1;
            const llama_seq_id seq = index + 1;
            if (!llama_memory_seq_rm(llama_get_memory(early), seq, -1, -1)) { return 2; }
            llama_memory_seq_cp(llama_get_memory(early), candidate.parent < 0 ? 0 : candidate.parent + 1, seq, -1, -1);
            early_batch.n_tokens = 0;
            batch_add_token_seqs(early_batch, candidate.token, base + depth, seq, -1, true);
            const auto decode_start = timer::now();
            if (llama_decode(early, early_batch) != 0) {
                std::fprintf(stderr, "gain tree early decode failed\n"); return 2;
            }
            const double measured = elapsed(decode_start);
            if (index > 0) {
                cost.draft_ms = 0.8 * cost.draft_ms + 0.2 * measured;
            }
            gain_tree_node node{candidate.parent, depth, candidate.token, candidate.probability, {}, {}, {}, {}};
            const float * hidden = llama_get_embeddings_layer_inp(early, opts.exit_layer);
            if (!hidden || !copy_logits(early, 0, n_vocab, node.logits)) { return 2; }
            node.hidden.assign(hidden, hidden + n_embd);
            nodes.push_back(std::move(node));
            if (candidate.parent >= 0) { nodes[candidate.parent].children.push_back(index); }
            leaves = best_leaves;
            gain += candidate.probability;
            if (depth + 1 < max_depth && (opts.ignore_eos || !llama_vocab_is_eog(vocab, candidate.token))) {
                add_frontier(index, nodes[index].logits);
            }
        }
        draft_ms += elapsed(phase);
        phase = timer::now();
        for (int i = 0; i < (int) nodes.size(); ++i) {
            if (!tail.memory_seq_rm(i + 1, -1, -1)) { return 2; }
            tail.memory_seq_cp(0, i + 1, -1, -1);
        }
        tail_batch.n_tokens = 0;
        if (pending) {
            std::vector<llama_seq_id> ids{0};
            for (int i = 0; i < (int) nodes.size(); ++i) { ids.push_back(i + 1); }
            batch_add_embd_seq_list(tail_batch, pending_hidden.data(), n_embd, base - 1, ids, true);
        }
        const int row_offset = pending ? 1 : 0;
        for (int i = 0; i < (int) nodes.size(); ++i) {
            std::vector<llama_seq_id> descendants;
            for (int j = i; j < (int) nodes.size(); ++j) {
                int ancestor = j;
                while (ancestor >= 0 && ancestor != i) { ancestor = nodes[ancestor].parent; }
                if (ancestor == i) { descendants.push_back(j + 1); }
            }
            batch_add_embd_seq_list(tail_batch, nodes[i].hidden.data(), n_embd, base + nodes[i].depth, descendants, true);
        }
        if(tail_batch.n_tokens!=(int)nodes.size()+row_offset)return 2;
        std::fprintf(stderr,"gain tile layout round=%d candidates=%zu pending=%d actual-rows=%d padded-hmx-rows=%d tiles32=%d leaves=%d\n",verifies+1,nodes.size(),row_offset,tail_batch.n_tokens,gain_tile_count(tail_batch.n_tokens,32)*32,gain_tile_count(tail_batch.n_tokens,32),leaves);
        if(opts.tail_budget_prefetch&&!tail.bind_graphless_budget_layout(tail_batch)){std::fprintf(stderr,"gain actual tree layout bind failed\n");return 2;}
        if (!tail.decode(tail_batch)) { std::fprintf(stderr, "gain tree target decode failed\n"); return 2; }
        std::vector<std::vector<float>> verified(tail_batch.n_tokens);
        for (int row = 0; row < tail_batch.n_tokens; ++row) {
            if (!tail.copy_logits(row, verified[row])) { return 2; }
        }
        const double tail_ms = elapsed(phase);
        verify_ms += tail_ms;
        cost.observe(tail_batch.n_tokens, leaves, tail_ms);
        ++verifies;
        drafted += (int) nodes.size();
        leaves_total += leaves;
        phase = timer::now();
        std::vector<float> current = pending ? verified[0] : target_logits;
        pending = false;
        int parent = -1, accepted = 0;
        bool leaf_end = false;
        while (generated_count < opts.n_predict) {
            const llama_token expected = argmax_token(current.data(), n_vocab);
            if (!opts.logits_prefix.empty() && !diagnostic.write(vocab, current.data(), n_vocab, generated_count, n_prompt + generated_count)) { return 2; }
            const auto & candidates = parent < 0 ? root_candidates : nodes[parent].candidates;
            if (!candidates.empty()) {
                const bool hit = std::find(candidates.begin(), candidates.end(), expected) != candidates.end();
                reliability = 0.98 * reliability + 0.02 * (hit ? 1.0 : 0.0);
                ++feedback;
            }
            int child = -1;
            for (int i = 0; i < (int) nodes.size(); ++i) {
                if (nodes[i].parent == parent && nodes[i].token == expected) { child = i; break; }
            }
            if (child < 0) {
                leaf_end = parent >= 0 && nodes[parent].children.empty();
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
        match_ms += elapsed(phase);
        std::fprintf(stderr, "gain tree round=%d nodes=%zu leaves=%d accepted=%d estimated-gain=%.3f verify-ms=%.3f reliability=%.3f\n",
                verifies, nodes.size(), leaves, accepted, gain, tail_ms, reliability);
        if (stop || generated_count >= opts.n_predict) { break; }
        phase = timer::now();
        const llama_seq_id active = parent < 0 ? 0 : parent + 1;
        const llama_pos next_pos = base + accepted;
        if (!llama_memory_seq_rm(llama_get_memory(early), active, next_pos, -1) ||
                !tail.memory_seq_rm(active, next_pos, -1)) { return 2; }
        if (active != 0) {
            llama_memory_seq_keep(llama_get_memory(early), active);
            llama_memory_seq_cp(llama_get_memory(early), active, 0, -1, -1);
            tail.memory_seq_keep(active);
            tail.memory_seq_cp(active, 0, -1, -1);
        }
        llama_memory_seq_keep(llama_get_memory(early), 0);
        tail.memory_seq_keep(0);
        rollback_ms += elapsed(phase);
        phase = timer::now();
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
            if (llama_decode(early, early_batch) != 0 || !copy_logits(early, -1, n_vocab, early_logits)) { return 2; }
            const float * hidden = llama_get_embeddings_layer_inp(early, opts.exit_layer);
            if (!hidden) { return 2; }
            pending_hidden.assign(hidden, hidden + n_embd);
            pending = true;
        }
        correction_ms += elapsed(phase);
    }
    const double seconds = elapsed(start) / 1000.0;
    if(opts.tail_budget_prefetch)tail.print_budget_read_stats();
    std::fprintf(stderr, "\nlayerskip generation %d tokens in %.3f seconds, speed: %.3f t/s\n", generated_count, seconds, generated_count / seconds);
    std::fprintf(stderr, "n_drafted = %d\nn_accept = %d\nn_verify = %d\nn_bonus = %d\n", drafted, accepted_total, verifies, bonus_count);
    std::fprintf(stderr, "accept = %.3f%%\nwindow accept = %.3f%%\n", drafted ? 100.0 * accepted_total / drafted : 0,
            verifies ? 100.0 * accepted_total / (verifies * opts.draft_max) : 0);
    std::fprintf(stderr, "tree nodes = %d\ntree leaves per verify = %.3f\ntree gain stops = %d\ntree calibration observations = %d\n",
            drafted, verifies ? (double) leaves_total / verifies : 0, gain_stops, feedback);
    print_token_trace(generated);
    std::fprintf(stderr, "phase prompt early = %.3f ms\nphase prompt tail = %.3f ms\nphase draft build = %.3f ms\nphase tail verify = %.3f ms\nphase correction early = %.3f ms\nphase match = %.3f ms\nphase KV rollback = %.3f ms\n",
            prompt_early_ms, prompt_tail_ms, draft_ms, verify_ms, correction_ms, match_ms, rollback_ms);
    std::fprintf(stderr, "tail HTP preload calls = %llu\ntail HTP preload failures = %llu\ntail HTP preload bytes = %llu\ntail HTP preload time = %.3f ms\n",
            (unsigned long long) tail.htp_preload_calls, (unsigned long long) tail.htp_preload_failures,
            (unsigned long long) tail.htp_preload_bytes, tail.htp_preload_time_us / 1000.0);
    return 0;
}
