#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__) || defined(__ANDROID__)
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#endif

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct seq_draft {
    bool active   = false;
    bool drafting = false;
    bool skip     = false;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;
    std::vector<std::vector<llama_token_data>> dists;
    std::vector<std::vector<llama_token_data>> diag_dists;
    std::vector<int> ranks;

    float path_prob = 1.0f;
    float gain_prob = 1.0f;

    struct common_sampler * smpl = nullptr;
};

struct tree_node {
    int parent = -1;
    int depth = 0;
    float path_prob = 1.0f;
};

struct tree_layout_stats {
    int nodes = 0;
    int leaves = 0;
    int max_depth = 0;
    int ancestor_pairs = 0;
};

struct tree_layout_plan {
    std::vector<int> signature;
    std::vector<int> order;
    std::vector<int> old_to_new;
    std::vector<std::vector<llama_seq_id>> seq_ids;
    tree_layout_stats stats;
};

struct tree_frontier_calibration {
    struct bucket {
        double hits = 0.0;
        double trials = 0.0;
    };

    std::vector<std::array<bucket, 4>> buckets;

    explicit tree_frontier_calibration(int max_depth) : buckets(max_depth) {}

    float estimate(int depth, int rank, float draft_probability) const {
        const int i_depth = std::min(std::max(depth - 1, 0), (int) buckets.size() - 1);
        const int i_rank = std::min(std::max(rank - 1, 0), 3);
        const bucket & value = buckets[i_depth][i_rank];
        constexpr double prior_weight = 8.0;
        return (float) ((value.hits + prior_weight * draft_probability) / (value.trials + prior_weight));
    }

    void observe(int depth, int rank, bool accepted) {
        if (depth <= 0 || buckets.empty()) {
            return;
        }
        const int i_depth = std::min(depth - 1, (int) buckets.size() - 1);
        const int i_rank = std::min(std::max(rank - 1, 0), 3);
        buckets[i_depth][i_rank].hits += accepted ? 1.0 : 0.0;
        buckets[i_depth][i_rank].trials += 1.0;
    }
};

struct spec_trace {
    struct event { const char * name; int lane; int os_tid; int round; int64_t start; int64_t stop; };
    const char * path = std::getenv("LLAMA_SPEC_TRACE");
    int64_t origin_mono_us = ggml_time_us();
    int64_t origin_boot_ns = []() {
#if defined(__linux__) || defined(__ANDROID__)
        struct timespec ts;
        clock_gettime(CLOCK_BOOTTIME, &ts);
        return (int64_t) ts.tv_sec * 1000000000LL + ts.tv_nsec;
#else
        return ggml_time_us() * 1000;
#endif
    }();
#if defined(__linux__) || defined(__ANDROID__)
    int os_pid = getpid();
#else
    int os_pid = 0;
#endif
    std::mutex mutex;
    std::vector<event> events;

    void add(const char * name, int lane, int round, int64_t start) {
        const int64_t stop = ggml_time_us();
        if (!path) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
#if defined(__linux__) || defined(__ANDROID__)
        const int os_tid = (int) syscall(__NR_gettid);
#else
        const int os_tid = (int) std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
        events.push_back({ name, lane, os_tid, round, start, stop });
    }

    ~spec_trace() {
        if (!path) {
            return;
        }
        FILE * f = std::fopen(path, "w");
        if (!f) {
            LOG_ERR("cannot write trace %s\n", path);
            return;
        }
        std::fprintf(f, "{\"traceEvents\":[{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":1,\"tid\":1,\"args\":{\"name\":\"CPU Draft / control\"}},{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":1,\"tid\":2,\"args\":{\"name\":\"HTP target call (host interval)\"}}]");
        std::fseek(f, -1, SEEK_CUR);
        for (const auto & e : events) {
            const int64_t boot_start_ns = origin_boot_ns + (e.start - origin_mono_us) * 1000;
            std::fprintf(f, ",{\"ph\":\"X\",\"pid\":1,\"tid\":%d,\"name\":\"%s\",\"ts\":%lld,\"dur\":%lld,\"args\":{\"round\":%d,\"os_pid\":%d,\"os_tid\":%d,\"boot_start_ns\":%lld}}", e.lane, e.name, (long long) (e.start - origin_mono_us), (long long) (e.stop - e.start), e.round, os_pid, e.os_tid, (long long) boot_start_ns);
        }
        std::fprintf(f, "],\"otherData\":{\"os_pid\":%d,\"origin_mono_us\":%lld,\"origin_boot_ns\":%lld}}\n", os_pid, (long long) origin_mono_us, (long long) origin_boot_ns);
        std::fclose(f);
    }
};

static std::vector<llama_token_data> greedy_tree_top_k(
        llama_context * ctx,
        int idx,
        int n_candidates,
        bool ignore_eos) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const float * logits = llama_get_logits_ith(ctx, idx);
    GGML_ASSERT(logits != nullptr);

    std::vector<llama_token_data> result;
    result.reserve(n_candidates);

    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token id = 0; id < n_vocab; ++id) {
        if (!std::isfinite(logits[id]) || (ignore_eos && llama_vocab_is_eog(vocab, id))) {
            continue;
        }

        const llama_token_data candidate = { id, logits[id], 0.0f };
        if ((int) result.size() < n_candidates) {
            result.push_back(candidate);
            continue;
        }

        auto worst = std::min_element(result.begin(), result.end(), [](const llama_token_data & a, const llama_token_data & b) {
            if (a.logit != b.logit) {
                return a.logit < b.logit;
            }
            return a.id > b.id;
        });
        if (candidate.logit > worst->logit || (candidate.logit == worst->logit && candidate.id < worst->id)) {
            *worst = candidate;
        }
    }

    std::sort(result.begin(), result.end(), [](const llama_token_data & a, const llama_token_data & b) {
        if (a.logit != b.logit) {
            return a.logit > b.logit;
        }
        return a.id < b.id;
    });

    if (!result.empty()) {
        double sum = 0.0;
        for (auto & candidate : result) {
            candidate.p = std::exp(candidate.logit - result.front().logit);
            sum += candidate.p;
        }
        for (auto & candidate : result) {
            candidate.p /= sum;
        }
    }

    return result;
}

static std::vector<int> tree_layout_signature(
        const std::vector<tree_node> & nodes,
        const std::vector<seq_draft> & drafts) {
    std::vector<int> result;
    result.reserve(2 * nodes.size() + drafts.size());
    for (const auto & node : nodes) {
        result.push_back(node.parent);
        result.push_back(node.depth);
    }
    for (const auto & draft : drafts) {
        result.push_back(draft.active ? (int) draft.i_batch_tgt.size() : -1);
        if (draft.active) {
            result.insert(result.end(), draft.i_batch_tgt.begin(), draft.i_batch_tgt.end());
        }
    }
    return result;
}

static tree_layout_stats tree_layout_dfs(
        llama_batch & batch,
        std::vector<tree_node> & nodes,
        std::vector<seq_draft> & drafts,
        tree_layout_plan & plan,
        bool & reused) {
    GGML_ASSERT(batch.n_tokens == (int) nodes.size());
    GGML_ASSERT(!nodes.empty());

    const std::vector<int> signature = tree_layout_signature(nodes, drafts);
    if (signature == plan.signature) {
        std::vector<llama_token> tokens(plan.order.size());
        std::vector<llama_pos> positions(plan.order.size());
        std::vector<int8_t> logits(plan.order.size());
        std::vector<tree_node> reordered(plan.order.size());

        for (int i = 0; i < (int) plan.order.size(); ++i) {
            const int old = plan.order[i];
            tokens[i] = batch.token[old];
            positions[i] = batch.pos[old];
            logits[i] = batch.logits[old];
            reordered[i] = nodes[old];
            reordered[i].parent = nodes[old].parent < 0 ? -1 : plan.old_to_new[nodes[old].parent];
        }

        for (auto & draft : drafts) {
            if (!draft.active) {
                continue;
            }
            for (int & idx : draft.i_batch_tgt) {
                idx = plan.old_to_new[idx];
                GGML_ASSERT(idx >= 0);
            }
        }

        for (int i = 0; i < (int) plan.order.size(); ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = positions[i];
            batch.logits[i] = logits[i];
            batch.n_seq_id[i] = plan.seq_ids[i].size();
            for (int j = 0; j < batch.n_seq_id[i]; ++j) {
                batch.seq_id[i][j] = plan.seq_ids[i][j];
            }
        }

        batch.n_tokens = plan.order.size();
        nodes = std::move(reordered);
        reused = true;
        return plan.stats;
    }

    reused = false;

    std::vector<bool> keep(nodes.size(), false);
    std::vector<int> leaves;

    for (int s = 0; s < (int) drafts.size(); ++s) {
        if (!drafts[s].active || drafts[s].i_batch_tgt.empty()) {
            continue;
        }

        leaves.push_back(s);
        for (const int idx : drafts[s].i_batch_tgt) {
            GGML_ASSERT(idx >= 0 && idx < (int) nodes.size());
            keep[idx] = true;
        }
    }

    keep[0] = true;

    std::vector<std::vector<int>> children(nodes.size());
    for (int i = 1; i < (int) nodes.size(); ++i) {
        if (!keep[i]) {
            continue;
        }
        GGML_ASSERT(nodes[i].parent >= 0 && nodes[i].parent < i);
        children[nodes[i].parent].push_back(i);
    }

    std::vector<int> order;
    order.reserve(nodes.size());
    std::function<void(int)> visit = [&](int node) {
        order.push_back(node);
        for (const int child : children[node]) {
            visit(child);
        }
    };
    visit(0);

    std::vector<int> old_to_new(nodes.size(), -1);
    for (int i = 0; i < (int) order.size(); ++i) {
        old_to_new[order[i]] = i;
    }

    std::vector<llama_token> tokens(order.size());
    std::vector<llama_pos> positions(order.size());
    std::vector<int8_t> logits(order.size());
    std::vector<std::vector<llama_seq_id>> seq_ids(order.size());
    std::vector<tree_node> reordered(order.size());

    for (int i = 0; i < (int) order.size(); ++i) {
        const int old = order[i];
        tokens[i] = batch.token[old];
        positions[i] = batch.pos[old];
        logits[i] = batch.logits[old];
        reordered[i] = nodes[old];
        reordered[i].parent = nodes[old].parent < 0 ? -1 : old_to_new[nodes[old].parent];
    }

    for (const int s : leaves) {
        for (int & idx : drafts[s].i_batch_tgt) {
            idx = old_to_new[idx];
            GGML_ASSERT(idx >= 0);
            seq_ids[idx].push_back(s);
        }
    }

    for (int i = 0; i < (int) order.size(); ++i) {
        GGML_ASSERT(!seq_ids[i].empty());

        batch.token[i] = tokens[i];
        batch.pos[i] = positions[i];
        batch.logits[i] = logits[i];
        batch.n_seq_id[i] = seq_ids[i].size();
        for (int j = 0; j < batch.n_seq_id[i]; ++j) {
            batch.seq_id[i][j] = seq_ids[i][j];
        }
    }

    batch.n_tokens = order.size();
    nodes = std::move(reordered);

    tree_layout_stats result;
    result.nodes = batch.n_tokens - 1;
    result.leaves = leaves.size();
    for (const auto & node : nodes) {
        result.max_depth = std::max(result.max_depth, node.depth);
    }

    for (int i = 0; i < batch.n_tokens; ++i) {
        GGML_ASSERT(nodes[i].parent < i);
        GGML_ASSERT(nodes[i].depth == (nodes[i].parent < 0 ? 0 : nodes[nodes[i].parent].depth + 1));

        int ancestor = i;
        while (ancestor >= 0) {
            bool shares_sequence = false;
            for (int si = 0; si < batch.n_seq_id[i] && !shares_sequence; ++si) {
                for (int sa = 0; sa < batch.n_seq_id[ancestor]; ++sa) {
                    shares_sequence = batch.seq_id[i][si] == batch.seq_id[ancestor][sa];
                    if (shares_sequence) {
                        break;
                    }
                }
            }

            GGML_ASSERT(shares_sequence);
            ++result.ancestor_pairs;
            ancestor = nodes[ancestor].parent;
        }
    }

    plan.signature = signature;
    plan.order = order;
    plan.old_to_new = old_to_new;
    plan.seq_ids = seq_ids;
    plan.stats = result;

    return result;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    // needed to get candidate probs even for temp <= 0.0
    params.sampling.n_probs = 128;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    if (params.speculative.draft.mparams.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // max number of parallel drafting sequences (i.e. tree branches)
    const int n_seq_dft = params.n_parallel;

    const auto & tree_widths = params.speculative.draft.tree_widths;
    const bool fixed_tree = !tree_widths.empty();
    const bool tree_frontier = params.speculative.draft.tree_frontier;

    if (fixed_tree && tree_frontier) {
        LOG_ERR("%s: --spec-tree-frontier cannot be combined with --spec-tree-widths\n", __func__);
        return 1;
    }

    int fixed_tree_nodes = 0;
    int fixed_tree_leaves = 1;
    for (const int width : tree_widths) {
        if (fixed_tree_leaves > std::numeric_limits<int>::max() / width) {
            LOG_ERR("%s: --spec-tree-widths is too large\n", __func__);
            return 1;
        }
        fixed_tree_leaves *= width;
        if (fixed_tree_nodes > std::numeric_limits<int>::max() - fixed_tree_leaves) {
            LOG_ERR("%s: --spec-tree-widths is too large\n", __func__);
            return 1;
        }
        fixed_tree_nodes += fixed_tree_leaves;
    }

    const int tree_max_depth = fixed_tree
        ? tree_widths.size()
        : params.speculative.draft.tree_max_depth > 0
            ? params.speculative.draft.tree_max_depth
            : params.speculative.draft.n_max;
    const int tree_max_nodes = fixed_tree
        ? fixed_tree_nodes
        : params.speculative.draft.tree_max_nodes > 0
            ? params.speculative.draft.tree_max_nodes
            : params.speculative.draft.n_max;
    const int tree_max_leaves = fixed_tree
        ? fixed_tree_leaves
        : params.speculative.draft.tree_max_leaves > 0
            ? params.speculative.draft.tree_max_leaves
            : n_seq_dft;

    if (tree_max_depth <= 0 || tree_max_nodes <= 0 || tree_max_leaves <= 0) {
        LOG_ERR("%s: token-tree limits must be positive\n", __func__);
        return 1;
    }

    if (tree_max_nodes + 1 > params.n_batch) {
        LOG_ERR("%s: --spec-tree-max-nodes + 1 must not exceed --batch-size\n", __func__);
        return 1;
    }

    if (tree_max_leaves > n_seq_dft) {
        LOG_ERR("%s: --spec-tree-max-leaves must not exceed --parallel\n", __func__);
        return 1;
    }

    if (n_seq_dft > 1) {
        params.kv_unified = true;
    }

    params.n_outputs_max = std::min(params.n_batch, tree_max_nodes + 1);
    params.n_outputs_max_per_seq = std::min(params.n_batch, tree_max_depth + 1);

    // probability threshold for splitting a draft branch (only for n_seq_dft > 1)
    const float p_draft_split = params.speculative.draft.p_split;
    const float tree_p_ratio = params.speculative.draft.tree_p_ratio;

    LOG_INF("token tree: max-depth %d, max-nodes %d, max-leaves %d\n",
            tree_max_depth, tree_max_nodes, tree_max_leaves);
    if (fixed_tree) {
        std::string widths;
        for (const int width : tree_widths) {
            widths += widths.empty() ? std::to_string(width) : "," + std::to_string(width);
        }
        LOG_INF("token tree: fixed maximum widths %s; side branches require path p >= %.3f and local ratio >= %.3f\n",
                widths.c_str(), p_draft_split, tree_p_ratio);
    }
    if (tree_frontier) {
        LOG_INF("token tree: calibrated frontier selection, minimum relative gain %.3f\n",
                params.speculative.draft.tree_frontier_min_gain);
    }

    std::default_random_engine rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sampling.seed);
    std::uniform_real_distribution<> u_dist;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    // load the draft model
    params.devices = params.speculative.draft.devices;
    params.model = params.speculative.draft.mparams;
    params.n_gpu_layers = params.speculative.draft.n_gpu_layers;
    params.n_outputs_max = params.n_parallel;
    params.n_outputs_max_per_seq = 1;
    if (params.speculative.draft.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.draft.cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.speculative.draft.cpuparams_batch.n_threads;
    params.tensor_buft_overrides     = params.speculative.draft.tensor_buft_overrides;

    auto llama_init_dft = common_init_from_params(params);

    model_dft = llama_init_dft->model();
    ctx_dft   = llama_init_dft->context();

    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_ERR("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return 1;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_ERR("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return 1;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_ERR("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return 1;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_ERR("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_ERR("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_ERR("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return 1;
            }
        }
    }

    auto * mem_tgt = llama_get_memory(ctx_tgt);
    auto * mem_dft = llama_get_memory(ctx_dft);

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    // eval the prompt with both models
    if ((n_input > 1 && llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), n_input - 1)) != 0) ||
        llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(), 1)) != 0 ||
        llama_decode(ctx_dft, llama_batch_get_one(inp.data(), n_input)) != 0) {
        LOG_ERR("%s: failed to decode the prompt\n", __func__);
        return 1;
    }

    const auto t_enc_end = ggml_time_us();

    // the 2 models should have the same vocab
    //GGML_ASSERT(n_vocab == llama_vocab_n_tokens(model_dft));

    int n_predict      = 0;
    int n_drafted      = 0;
    int n_accept       = 0;
    int n_verify       = 0;
    int n_verify_nodes = 0;

    int64_t t_verify_us = 0;
    int64_t t_accept_us = 0;
    int64_t t_target_kv_select_us = 0;
    int64_t t_draft_sync_us = 0;
    int64_t t_tree_build_us = 0;
    int64_t t_draft_sample_us = 0;
    int64_t t_draft_kv_branch_us = 0;
    int64_t t_draft_decode_us = 0;
    int64_t t_tree_layout_us = 0;
    int64_t t_target_kv_prepare_us = 0;
    int64_t t_target_enqueue_us = 0;
    int64_t t_target_sync_us = 0;
    int64_t t_draft_enqueue_us = 0;
    int64_t t_draft_decode_sync_us = 0;
    int64_t n_tree_nodes = 0;
    int64_t n_tree_leaves = 0;
    int64_t n_tree_depth = 0;
    int64_t n_tree_ancestor_pairs = 0;
    int64_t n_verify_padding = 0;
    int64_t n_tree_layout_reuse = 0;
    int64_t n_tree_layout_build = 0;
    int64_t n_tree_draft_decode_calls = 0;
    int64_t n_tree_draft_decode_nodes = 0;
    int64_t n_tree_pruned_side_candidates = 0;
    int64_t n_tree_pruned_ratio_candidates = 0;
    int64_t n_tree_frontier_candidates = 0;
    int64_t n_tree_frontier_selected = 0;
    int64_t n_tree_frontier_pruned = 0;
    int64_t n_tree_frontier_stopped_leaves = 0;
    int64_t n_side_branch_entries = 0;
    int64_t n_side_branch_accepted = 0;
    int64_t n_target_rank_1 = 0;
    int64_t n_target_rank_2 = 0;
    int64_t n_target_rank_3 = 0;
    int64_t n_target_rank_miss = 0;
    int64_t n_side_accept_len_1 = 0;
    int64_t n_side_accept_len_2 = 0;
    int64_t n_side_accept_len_3 = 0;
    int64_t n_side_accept_len_4_plus = 0;

    tree_frontier_calibration frontier_calibration(tree_max_depth);

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    // used to determine end of generation
    bool has_eos = false;

    // target model sampling context (reuse the llama_context's sampling instance)
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    const int greedy_tree_top_k_count = fixed_tree
        ? *std::max_element(tree_widths.begin(), tree_widths.end())
        : tree_max_leaves;
    const bool greedy_tree = params.sampling.temp <= 0.0f && greedy_tree_top_k_count > 1;
    if (greedy_tree) {
        LOG_INF("greedy target with raw draft top-%d tree candidates\n", greedy_tree_top_k_count);
    }

    // draft sequence data
    std::vector<seq_draft> drafts(n_seq_dft);

    for (int s = 0; s < n_seq_dft; ++s) {
        // allocate llama_sampler for each draft sequence
        drafts[s].smpl = common_sampler_init(model_dft, params.sampling);
    }

    llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);
    tree_layout_plan layout_plan;

    spec_trace trace;
    const bool pipeline = std::getenv("LLAMA_SPEC_PIPELINE") && std::strcmp(std::getenv("LLAMA_SPEC_PIPELINE"), "1") == 0;
    const bool tree_diag = std::getenv("LLAMA_SPEC_DIAG") && std::strcmp(std::getenv("LLAMA_SPEC_DIAG"), "1") == 0;
    LOG_INF("pipeline final draft decode = %d\n", pipeline);
    LOG_INF("tree decision diagnostics = %d\n", tree_diag);
    std::future<int> pending_draft;

    const auto t_dec_start = ggml_time_us();

    // sample from the last token of the prompt
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    while (true) {
        const int64_t t_accept_start = ggml_time_us();
        std::set<int> active_seqs = {};
        bool on_side_branch = false;
        int side_branch_accept_len = 0;
        int side_branch_entry_depth = -1;
        int side_branch_entry_rank = 0;
        bool diag_last_had_candidate = false;
        int diag_last_target_rank = 0;

        // print current draft sequences
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            active_seqs.insert(s);
            const auto & tokens = drafts[s].tokens;

            LOG_DBG("draft %d: %s\n", s, string_from(ctx_dft, tokens).c_str());
        }

        int i_dft  = 0;
        int s_keep = 0;

        llama_token token_id;
        std::string token_str;

        // loop until we fail to accept a drafted token or we run out of drafted tokens
        while (true) {
            std::vector<std::pair<llama_token, int>> frontier_observations;
            if (tree_frontier) {
                std::set<std::pair<int, llama_token>> seen;
                for (const int s : active_seqs) {
                    if (i_dft >= (int) drafts[s].tokens.size() || i_dft >= (int) drafts[s].ranks.size()) {
                        continue;
                    }
                    const auto key = std::make_pair(drafts[s].i_batch_tgt[i_dft], drafts[s].tokens[i_dft]);
                    if (seen.insert(key).second) {
                        frontier_observations.emplace_back(drafts[s].tokens[i_dft], drafts[s].ranks[i_dft]);
                    }
                }
            }

            // check if the target token matches any of the drafts
            // for stochastic sampling, attempt to match the token with the drafted tokens
            {
                bool accept = false;
                if (params.sampling.temp > 0) {
                    // stochastic verification
                    common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft], true);

                    auto & dist_tgt = *common_sampler_get_candidates(smpl, true);

                    float p_tgt = 0.0f;
                    float p_dft = 0.0f;

                    while (active_seqs.size() > 0) {
                        // randomly select a sequence to verify from active sequences
                        std::uniform_int_distribution<unsigned int> u_int_dist(0, active_seqs.size() - 1);
                        int s = *std::next(active_seqs.begin(), u_int_dist(rng));
                        if (i_dft >= (int) drafts[s].tokens.size()) {
                            drafts[s].active = false;
                            active_seqs.erase(s);
                            continue;
                        }
                        if (accept) {
                            // if we already accepted a token, we can skip the rest
                            if (drafts[s].tokens[i_dft] != drafts[s_keep].tokens[i_dft]) {
                                drafts[s].active = false;
                                active_seqs.erase(s);
                            }
                            continue;
                        }

                        p_tgt = 0.0f;
                        p_dft = 0.0f;

                        LOG_DBG("verifying sequence #%d at pos #%d from %d active sequence(s)\n", s, i_dft, (int) active_seqs.size());
                        float r = u_dist(rng);
                        llama_token_data_array dist_dft = { drafts[s].dists[i_dft].data() , drafts[s].dists[i_dft].size(), LLAMA_TOKEN_NULL, true };

                        //GGML_ASSERT(dist_tgt.size <= dist_dft.size);

                        // acquire the token probabilities assigned by the draft and target models
                        for (size_t i = 0; i < dist_tgt.size; i++) {
                            if (dist_tgt.data[i].id == drafts[s].tokens[i_dft]) {
                                p_tgt = dist_tgt.data[i].p;
                                break;
                            }
                        }
                        for (size_t i = 0; i < dist_dft.size; i++) {
                            if (dist_dft.data[i].id == drafts[s].tokens[i_dft]) {
                                p_dft = dist_dft.data[i].p;
                                break;
                            }
                        }
                        LOG_DBG("r = %f, p_dft = %f, p_tgt = %f\n", r, p_dft, p_tgt);
                        if (p_dft > 0.0f && r <= p_tgt / p_dft) {
                            s_keep = s;
                            accept = true;
                            token_id = drafts[s].tokens[i_dft];
                            token_str = common_token_to_piece(ctx_tgt, token_id);
                            common_sampler_accept(smpl, token_id, true);

                            LOG_DBG("draft token %d of sequence %d (%d, '%s') accepted\n", i_dft, s, token_id, token_str.c_str());
                            break;
                        } else {
                            LOG_DBG("draft token %d of sequence %d (%d, '%s') rejected\n", i_dft, s, drafts[s].tokens[i_dft], common_token_to_piece(ctx_tgt, drafts[s].tokens[i_dft]).c_str());
                            drafts[s].active = false;

                            // calculate residual probability
                            GGML_ASSERT(dist_tgt.sorted);
                            GGML_ASSERT(dist_dft.sorted);

                            // sort dist by id
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });
                            std::sort(dist_dft.data, dist_dft.data + dist_dft.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });

                            float sum_probs = 0.0f;
                            size_t i_dft_prob = 0;

                            for (size_t i = 0; i < dist_tgt.size; ++i) {
                                while (i_dft_prob < dist_dft.size && dist_dft.data[i_dft_prob].id < dist_tgt.data[i].id) {
                                    ++i_dft_prob;
                                }

                                const float p_draft = i_dft_prob < dist_dft.size && dist_dft.data[i_dft_prob].id == dist_tgt.data[i].id
                                    ? dist_dft.data[i_dft_prob].p
                                    : 0.0f;

                                dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p - p_draft);
                                sum_probs += dist_tgt.data[i].p;
                            }

                            if (sum_probs > 0.0f) {
                                for (size_t i = 0; i < dist_tgt.size; ++i) {
                                    dist_tgt.data[i].p /= sum_probs;
                                }
                            }

                            // sort dist_tgt by p desc
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.p > b.p;
                            });
                        }

                        active_seqs.erase(s);
                        for (int i = 0; i < n_seq_dft; i++) {
                            if (i == s) {
                                continue;
                            }
                            if (drafts[i].active && drafts[i].tokens[i_dft] == drafts[s].tokens[i_dft]) {
                                // synchronize active status for sequences with the same drafted token
                                drafts[i].active = drafts[i].active && accept;
                                if (!drafts[i].active) {
                                    active_seqs.erase(i);
                                }
                            }
                        }
                    }

                    if (!accept) {
                        // all drafted tokens were rejected
                        // sample from the target model
                        LOG_DBG("all drafted tokens were rejected, sampling from residual distribution\n");
                        std::vector<float> probs(dist_tgt.size);
                        for (size_t i = 0; i < dist_tgt.size; ++i) {
                            probs[i] = dist_tgt.data[i].p;
                        }

                        std::discrete_distribution<> dist(probs.begin(), probs.end());

                        const int idx = dist(rng);

                        token_id = dist_tgt.data[idx].id;
                        common_sampler_accept(smpl, token_id, true);
                        token_str = common_token_to_piece(ctx_tgt, token_id);
                    }
                } else {
                    // greedy verification

                    // sample from the target model
                    LOG_DBG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);
                    token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]);

                    common_sampler_accept(smpl, token_id, true);

                    token_str = common_token_to_piece(ctx_tgt, token_id);

                    if (tree_diag && n_verify > 0) {
                        int best_rank = 0;
                        float top_p[3] = { 0.0f, 0.0f, 0.0f };
                        bool have_dist = false;
                        diag_last_had_candidate = false;

                        for (int s = 0; s < n_seq_dft; ++s) {
                            if (!drafts[s].active || i_dft >= (int) drafts[s].tokens.size()) {
                                continue;
                            }
                            diag_last_had_candidate = true;
                            if (i_dft >= (int) drafts[s].diag_dists.size()) {
                                continue;
                            }

                            const auto & dist = drafts[s].diag_dists[i_dft];
                            if (!have_dist) {
                                for (int k = 0; k < std::min(3, (int) dist.size()); ++k) {
                                    top_p[k] = dist[k].p;
                                }
                                have_dist = true;
                            }
                            for (int k = 0; k < std::min(3, (int) dist.size()); ++k) {
                                if (dist[k].id == token_id && (best_rank == 0 || k + 1 < best_rank)) {
                                    best_rank = k + 1;
                                }
                            }
                        }

                        diag_last_target_rank = best_rank;
                        if (diag_last_had_candidate) {
                            if (best_rank == 1) {
                                ++n_target_rank_1;
                            } else if (best_rank == 2) {
                                ++n_target_rank_2;
                            } else if (best_rank == 3) {
                                ++n_target_rank_3;
                            } else {
                                ++n_target_rank_miss;
                            }
                            LOG_INF("tree diag decision: round=%d depth=%d target=%d rank=%d top3-p=%.6f,%.6f,%.6f\n",
                                    n_verify, i_dft + 1, token_id, best_rank, top_p[0], top_p[1], top_p[2]);
                        }
                    }

                    for (int s = 0; s < n_seq_dft; ++s) {
                        if (!drafts[s].active) {
                            continue;
                        }

                        if (i_dft < (int) drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                            LOG_DBG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, token_id, token_str.c_str());

                            s_keep = s;
                            accept = true;
                        } else {
                            drafts[s].active = false;
                        }
                    }
                }

                if (tree_frontier) {
                    for (const auto & observation : frontier_observations) {
                        frontier_calibration.observe(i_dft + 1, observation.second,
                                accept && observation.first == token_id);
                    }
                }

                if (llama_vocab_is_eog(vocab_tgt, token_id)) {
                    has_eos = true;
                }
                ++n_predict;

                if (accept) {
                    if (!on_side_branch && s_keep != 0 && i_dft < (int) drafts[0].tokens.size() && token_id != drafts[0].tokens[i_dft]) {
                        on_side_branch = true;
                        side_branch_entry_depth = i_dft + 1;
                        side_branch_entry_rank = diag_last_target_rank;
                        ++n_side_branch_entries;
                    }
                    if (on_side_branch) {
                        ++side_branch_accept_len;
                        ++n_side_branch_accepted;
                    }
                    ++n_accept;
                    ++n_past_tgt;
                    ++n_past_dft;
                    ++i_dft;
                    if (params.use_color) {
                        // Color token according to its origin sequence
                        LOG("\u001b[%dm%s\u001b[37m", (36 - s_keep % 6), token_str.c_str());
                    } else {
                        LOG("%s", token_str.c_str());
                    }
                    continue;
                } else {
                    LOG("%s", token_str.c_str());
                    break;
                }
            }
        }

        if (tree_diag && n_verify > 0) {
            if (on_side_branch) {
                if (side_branch_accept_len == 1) {
                    ++n_side_accept_len_1;
                } else if (side_branch_accept_len == 2) {
                    ++n_side_accept_len_2;
                } else if (side_branch_accept_len == 3) {
                    ++n_side_accept_len_3;
                } else if (side_branch_accept_len >= 4) {
                    ++n_side_accept_len_4_plus;
                }
            }
            LOG_INF("tree diag outcome: round=%d accepted=%d side=%d side-rank=%d branch-depth=%d side-accepted=%d stop=%s target=%d\n",
                    n_verify, i_dft, on_side_branch, side_branch_entry_rank, side_branch_entry_depth,
                    side_branch_accept_len, diag_last_had_candidate ? "mismatch" : "tree-exhausted", token_id);
        }

        t_accept_us += ggml_time_us() - t_accept_start;
        trace.add("accept_and_sample", 1, n_verify, t_accept_start);

        {
            LOG_DBG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", token_id, token_str.c_str());

            const int64_t t_target_kv_select_start = ggml_time_us();

            // TODO: simplify
            {
                LOG_DBG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", s_keep, n_past_tgt, n_past_dft);

                if (s_keep != 0) {
                    llama_memory_seq_keep(mem_dft, s_keep);
                    llama_memory_seq_cp  (mem_dft, s_keep, 0, -1, -1);
                }
                llama_memory_seq_keep(mem_dft, 0);

                llama_memory_seq_rm  (mem_tgt, s_keep, n_past_tgt, -1);
                if (s_keep != 0) {
                    llama_memory_seq_keep(mem_tgt, s_keep);
                    llama_memory_seq_cp  (mem_tgt, s_keep, 0, -1, -1);
                }
                llama_memory_seq_keep(mem_tgt, 0);
            }

            t_target_kv_select_us += ggml_time_us() - t_target_kv_select_start;
            trace.add("KV_select", 1, n_verify, t_target_kv_select_start);

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                drafts[s].tokens.clear();
                drafts[s].i_batch_tgt.clear();
                drafts[s].dists.clear();
                drafts[s].diag_dists.clear();
                drafts[s].ranks.clear();
            }
            // note: will be erased after the speculation phase
            drafts[0].tokens.push_back(token_id);
            drafts[0].dists.push_back(std::vector<llama_token_data>());
            drafts[0].diag_dists.push_back(std::vector<llama_token_data>());
            drafts[0].ranks.push_back(0);
            drafts[0].i_batch_tgt.push_back(0);

            common_batch_clear(batch_dft);
            common_batch_add  (batch_dft, token_id, n_past_dft, { 0 }, true);

            llama_memory_seq_rm(mem_dft, 0, n_past_dft, -1);
            // LOG_DBG("dft batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_dft, batch_dft).c_str());
            const int64_t t_draft_sync_start = ggml_time_us();
            if (llama_decode(ctx_dft, batch_dft) != 0) {
                LOG_ERR("%s: failed to synchronize the draft context\n", __func__);
                return 1;
            }
            llama_synchronize(ctx_dft);
            t_draft_sync_us += ggml_time_us() - t_draft_sync_start;
            trace.add("draft_seed_decode", 1, n_verify, t_draft_sync_start);

            ++n_past_dft;
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        if (drafts[0].smpl) {
            common_sampler_free(drafts[0].smpl);
        }
        drafts[0].smpl = common_sampler_clone(smpl);

        const int64_t t_tree_build_start = ggml_time_us();
        const int64_t t_round_draft_decode_start = t_draft_decode_us;

        int n_seq_cur  = 1;
        int n_past_cur = n_past_dft;

        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].active   = false;
            drafts[s].drafting = false;
        }
        drafts[0].active      = true;
        drafts[0].drafting    = true;
        drafts[0].i_batch_dft = 0;
        drafts[0].path_prob   = 1.0f;
        drafts[0].gain_prob   = 1.0f;

        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

        std::vector<tree_node> tree_nodes(1);

        // sample candidate nodes from the draft model
        for (int i = 0; i < tree_max_depth; ++i) {
            struct tree_candidate {
                int src;
                int rank;
                llama_token token;
                float score;
                float gain;
                std::vector<llama_token_data> dist;
                std::vector<llama_token_data> diag_dist;
            };

            std::vector<tree_candidate> candidates;

            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting) {
                    continue;
                }

                const int64_t t_draft_sample_start = ggml_time_us();
                std::vector<llama_token_data> greedy_candidates;
                llama_token_data_array greedy_candidates_view = {};
                const llama_token_data_array * cur_p;

                if (greedy_tree) {
                    const int n_requested = fixed_tree ? tree_widths[i] : tree_max_leaves + 3;
                    greedy_candidates = greedy_tree_top_k(
                            ctx_dft, drafts[s].i_batch_dft, n_requested, params.sampling.ignore_eos);
                    greedy_candidates_view = {
                        greedy_candidates.data(), greedy_candidates.size(), 0, true,
                    };
                    cur_p = &greedy_candidates_view;
                } else {
                    common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);
                    cur_p = common_sampler_get_candidates(drafts[s].smpl, true);
                }

                const int n_candidates = fixed_tree
                    ? std::min((int) tree_widths[i], (int) cur_p->size)
                    : std::min(tree_max_leaves + 3, (int) cur_p->size);

                std::vector<llama_token_data> diag_dist;
                if (tree_diag) {
                    if (params.sampling.temp <= 0.0f) {
                        diag_dist = greedy_tree_top_k(ctx_dft, drafts[s].i_batch_dft, 3, params.sampling.ignore_eos);
                    } else {
                        diag_dist.assign(cur_p->data, cur_p->data + std::min(3, (int) cur_p->size));
                    }
                }

                for (int k = 0; k < n_candidates; ++k) {
                    const float path_score = drafts[s].path_prob * cur_p->data[k].p;
                    const float calibrated_probability = tree_frontier
                        ? frontier_calibration.estimate(i + 1, k + 1, cur_p->data[k].p)
                        : cur_p->data[k].p;
                    const float path_gain = drafts[s].gain_prob * calibrated_probability;
                    const float local_ratio = cur_p->data[0].p > 0.0f ? cur_p->data[k].p / cur_p->data[0].p : 0.0f;

                    LOG_DBG(" - draft candidate %3d for seq %3d, pos %3d: %6d (p %8.3f, ratio %8.3f, path %8.3f) '%s'\n",
                            k, s, i, cur_p->data[k].id, cur_p->data[k].p, local_ratio, path_score,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());

                    if (k > 0 && local_ratio < tree_p_ratio) {
                        n_tree_pruned_side_candidates += n_candidates - k;
                        n_tree_pruned_ratio_candidates += n_candidates - k;
                        break;
                    }

                    if ((s != 0 || k > 0) && path_score < p_draft_split) {
                        n_tree_pruned_side_candidates += n_candidates - k;
                        break;
                    }

                    candidates.push_back({
                        /* .src   = */ s,
                        /* .rank  = */ k + 1,
                        /* .token = */ cur_p->data[k].id,
                        /* .score = */ path_score,
                        /* .gain  = */ path_gain,
                        /* .dist  = */ { cur_p->data, cur_p->data + cur_p->size },
                        /* .diag_dist = */ diag_dist,
                    });
                }
                t_draft_sample_us += ggml_time_us() - t_draft_sample_start;
                trace.add("draft_candidates", 1, n_verify + 1, t_draft_sample_start);
            }

            if (tree_frontier) {
                n_tree_frontier_candidates += candidates.size();

                const int n_remaining = tree_max_nodes - (batch_tgt.n_tokens - 1);
                const int n_active = std::count_if(drafts.begin(), drafts.end(), [](const seq_draft & draft) {
                    return draft.active;
                });
                int n_extra_leaves = 0;
                std::set<int> selected_sources;
                std::vector<tree_candidate> selected;
                selected.reserve(std::min((int) candidates.size(), n_remaining));
                float best_rate = 0.0f;

                while ((int) selected.size() < n_remaining) {
                    int i_best = -1;
                    float candidate_best_rate = -1.0f;

                    for (int j = 0; j < (int) candidates.size(); ++j) {
                        const auto & candidate = candidates[j];
                        if (candidate.src < 0) {
                            continue;
                        }

                        const bool creates_leaf = selected_sources.count(candidate.src) > 0;
                        if (creates_leaf && n_active + n_extra_leaves >= tree_max_leaves) {
                            continue;
                        }

                        const float draft_cost = creates_leaf ? 2.0f : 1.0f;
                        const float rate = candidate.gain / draft_cost;
                        if (rate > candidate_best_rate) {
                            candidate_best_rate = rate;
                            i_best = j;
                        }
                    }

                    if (i_best < 0) {
                        break;
                    }
                    if (selected.empty()) {
                        best_rate = candidate_best_rate;
                    } else if (candidate_best_rate < best_rate * params.speculative.draft.tree_frontier_min_gain) {
                        break;
                    }

                    tree_candidate candidate = std::move(candidates[i_best]);
                    candidates[i_best].src = -1;
                    if (!selected_sources.insert(candidate.src).second) {
                        ++n_extra_leaves;
                    }
                    selected.push_back(std::move(candidate));
                }

                n_tree_frontier_selected += selected.size();
                n_tree_frontier_pruned += candidates.size() - selected.size();
                candidates = std::move(selected);
            } else if (!fixed_tree) {
                std::sort(candidates.begin(), candidates.end(), [](const tree_candidate & a, const tree_candidate & b) {
                    if (a.score != b.score) {
                        return a.score > b.score;
                    }
                    if (a.src != b.src) {
                        return a.src < b.src;
                    }
                    return a.token < b.token;
                });

                const int n_remaining = tree_max_nodes - (batch_tgt.n_tokens - 1);
                if ((int) candidates.size() > std::min(tree_max_leaves, n_remaining)) {
                    candidates.resize(std::min(tree_max_leaves, n_remaining));
                }
            }

            if (candidates.empty()) {
                break;
            }

            std::set<int> selected_sources;
            for (const auto & candidate : candidates) {
                selected_sources.insert(candidate.src);
            }
            std::vector<bool> was_active(n_seq_dft, false);
            for (int s = 0; s < n_seq_dft; ++s) {
                was_active[s] = drafts[s].active;
            }

            struct tree_expansion {
                int src;
                int dst;
                int rank;
                llama_token token;
                float score;
                float gain;
                std::vector<llama_token> tokens;
                std::vector<std::vector<llama_token_data>> dists;
                std::vector<std::vector<llama_token_data>> diag_dists;
                std::vector<int> ranks;
                std::vector<int> i_batch_tgt;
                std::vector<llama_token_data> dist;
                std::vector<llama_token_data> diag_dist;
                common_sampler * smpl;
            };

            std::vector<tree_expansion> expansions;
            std::set<int> reused_sources;
            std::set<int> used_destinations;

            for (const auto & candidate : candidates) {
                int dst = -1;

                if (reused_sources.insert(candidate.src).second) {
                    dst = candidate.src;
                } else {
                    for (int s = 0; s < tree_max_leaves; ++s) {
                        const bool destination_available = tree_frontier
                            ? !was_active[s]
                            : selected_sources.count(s) == 0;
                        if (destination_available && used_destinations.count(s) == 0) {
                            dst = s;
                            break;
                        }
                    }
                }

                GGML_ASSERT(dst >= 0);
                used_destinations.insert(dst);

                expansions.push_back({
                    /* .src         = */ candidate.src,
                    /* .dst         = */ dst,
                    /* .rank        = */ candidate.rank,
                    /* .token       = */ candidate.token,
                    /* .score       = */ candidate.score,
                    /* .gain        = */ candidate.gain,
                    /* .tokens      = */ drafts[candidate.src].tokens,
                    /* .dists       = */ drafts[candidate.src].dists,
                    /* .diag_dists  = */ drafts[candidate.src].diag_dists,
                    /* .ranks       = */ drafts[candidate.src].ranks,
                    /* .i_batch_tgt = */ drafts[candidate.src].i_batch_tgt,
                    /* .dist        = */ candidate.dist,
                    /* .diag_dist   = */ candidate.diag_dist,
                    /* .smpl        = */ common_sampler_clone(drafts[candidate.src].smpl),
                });
            }

            const int64_t t_draft_kv_branch_start = ggml_time_us();
            for (const auto & expansion : expansions) {
                if (expansion.dst == expansion.src) {
                    continue;
                }

                LOG_DBG("splitting seq %3d into %3d with path probability %.6f\n",
                        expansion.src, expansion.dst, expansion.score);

                llama_memory_seq_rm(mem_dft, expansion.dst, -1, -1);
                llama_memory_seq_cp(mem_dft, expansion.src, expansion.dst, -1, -1);

                for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                    int n_seq_id = 0;
                    bool has_src = false;

                    for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                        const int seq_id = batch_tgt.seq_id[t][p];
                        has_src = has_src || seq_id == expansion.src;
                        if (seq_id != expansion.dst) {
                            batch_tgt.seq_id[t][n_seq_id++] = seq_id;
                        }
                    }

                    batch_tgt.n_seq_id[t] = n_seq_id;
                    if (has_src) {
                        batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]++] = expansion.dst;
                    }
                }
            }
            t_draft_kv_branch_us += ggml_time_us() - t_draft_kv_branch_start;

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                drafts[s].drafting = false;
            }
            if (tree_frontier) {
                for (int s = 0; s < n_seq_dft; ++s) {
                    if (was_active[s] && selected_sources.count(s) == 0) {
                        drafts[s].active = true;
                        ++n_tree_frontier_stopped_leaves;
                    }
                }
            }

            common_batch_clear(batch_dft);

            for (auto & expansion : expansions) {
                auto & draft = drafts[expansion.dst];

                if (draft.smpl) {
                    common_sampler_free(draft.smpl);
                }

                draft.active = true;
                draft.drafting = true;
                draft.tokens = std::move(expansion.tokens);
                draft.dists = std::move(expansion.dists);
                draft.diag_dists = std::move(expansion.diag_dists);
                draft.ranks = std::move(expansion.ranks);
                draft.i_batch_tgt = std::move(expansion.i_batch_tgt);
                draft.path_prob = expansion.score;
                draft.gain_prob = expansion.gain;
                draft.smpl = expansion.smpl;

                common_sampler_accept(draft.smpl, expansion.token, true);

                draft.tokens.push_back(expansion.token);
                draft.dists.push_back(std::move(expansion.dist));
                draft.diag_dists.push_back(std::move(expansion.diag_dist));
                draft.ranks.push_back(expansion.rank);
                const int parent = draft.i_batch_tgt.back();
                draft.i_batch_tgt.push_back(batch_tgt.n_tokens);

                common_batch_add(batch_tgt, expansion.token, n_past_tgt + i + 1, { expansion.dst }, true);
                tree_nodes.push_back({ parent, i + 1, expansion.score });

                draft.i_batch_dft = batch_dft.n_tokens;
                common_batch_add(batch_dft, expansion.token, n_past_cur, { expansion.dst }, true);
            }

            n_seq_cur = std::count_if(drafts.begin(), drafts.end(), [](const seq_draft & draft) {
                return draft.active;
            });

            // no sequence is drafting anymore
            if (batch_dft.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            const bool last_level = i + 1 == tree_max_depth || batch_tgt.n_tokens - 1 >= tree_max_nodes;
            if (pipeline && last_level) {
                const int round = n_verify + 1;
                pending_draft = std::async(std::launch::async, [&, round]() {
                    const int64_t start = ggml_time_us();
                    const int result = llama_decode(ctx_dft, batch_dft);
                    llama_synchronize(ctx_dft);
                    t_draft_decode_us += ggml_time_us() - start;
                    trace.add("draft_final_decode_async", 1, round, start);
                    return result;
                });
                ++n_tree_draft_decode_calls;
                n_tree_draft_decode_nodes += batch_dft.n_tokens;
                break;
            }
            const int64_t t_draft_decode_start = ggml_time_us();
            const int64_t t_draft_enqueue_start = ggml_time_us();
            if (llama_decode(ctx_dft, batch_dft) != 0) {
                LOG_ERR("%s: failed to decode the draft batch\n", __func__);
                return 1;
            }
            t_draft_enqueue_us += ggml_time_us() - t_draft_enqueue_start;

            const int64_t t_draft_sync_start = ggml_time_us();
            llama_synchronize(ctx_dft);
            t_draft_decode_sync_us += ggml_time_us() - t_draft_sync_start;
            t_draft_decode_us += ggml_time_us() - t_draft_decode_start;
            trace.add("draft_decode", 1, n_verify + 1, t_draft_decode_start);
            ++n_tree_draft_decode_calls;
            n_tree_draft_decode_nodes += batch_dft.n_tokens;
            ++n_past_cur;

            if (batch_tgt.n_tokens - 1 >= tree_max_nodes) {
                break;
            }
        }

        t_tree_build_us += ggml_time_us() - t_tree_build_start;
        const int64_t t_round_tree_build_us = ggml_time_us() - t_tree_build_start;
        const int64_t t_round_draft_decode_us = t_draft_decode_us - t_round_draft_decode_start;

        const int64_t t_tree_layout_start = ggml_time_us();
        bool layout_reused = false;
        const tree_layout_stats layout = tree_layout_dfs(batch_tgt, tree_nodes, drafts, layout_plan, layout_reused);
        layout_reused ? ++n_tree_layout_reuse : ++n_tree_layout_build;
        t_tree_layout_us += ggml_time_us() - t_tree_layout_start;
        trace.add("tree_layout", 1, n_verify + 1, t_tree_layout_start);
        n_seq_cur = layout.leaves;
        n_tree_nodes += layout.nodes;
        n_tree_leaves += layout.leaves;
        n_tree_depth += layout.max_depth;
        n_tree_ancestor_pairs += layout.ancestor_pairs;

        LOG_DBG("tree layout: nodes = %d, leaves = %d, max depth = %d\n",
                layout.nodes, layout.leaves, layout.max_depth);

        const int n_nodes = layout.nodes;
        if (params.speculative.draft.tree_pad_nodes) {
            const int n_target_tokens = tree_max_nodes + 1;
            GGML_ASSERT(batch_tgt.n_tokens <= n_target_tokens);

            const llama_token pad_token = batch_tgt.token[batch_tgt.n_tokens - 1];
            llama_pos pad_pos = batch_tgt.pos[batch_tgt.n_tokens - 1] + 1;

            while (batch_tgt.n_tokens < n_target_tokens) {
                common_batch_add(batch_tgt, pad_token, pad_pos++, { 0 }, true);

                const int i_pad = batch_tgt.n_tokens - 1;
                batch_tgt.n_seq_id[i_pad] = n_seq_dft;
                for (int s = 0; s < n_seq_dft; ++s) {
                    batch_tgt.seq_id[i_pad][s] = s;
                }

                ++n_verify_padding;
            }
        }

        // evaluate the target model on the drafted tokens
        {
            const int64_t t_target_kv_prepare_start = ggml_time_us();
            for (int s = 1; s < n_seq_dft; ++s) {
                if (drafts[s].active) {
                    llama_memory_seq_cp(mem_tgt, 0, s, -1, -1);
                }
            }
            t_target_kv_prepare_us += ggml_time_us() - t_target_kv_prepare_start;
            trace.add("target_KV_prepare", 1, n_verify + 1, t_target_kv_prepare_start);

            const int64_t t_verify_start = ggml_time_us();
            const int64_t t_target_enqueue_start = ggml_time_us();

            // LOG_DBG("target batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str());
            if (llama_decode(ctx_tgt, batch_tgt) != 0) {
                LOG_ERR("%s: failed to decode the target tree batch\n", __func__);
                return 1;
            }
            t_target_enqueue_us += ggml_time_us() - t_target_enqueue_start;

            const int64_t t_target_sync_start = ggml_time_us();
            llama_synchronize(ctx_tgt);
            t_target_sync_us += ggml_time_us() - t_target_sync_start;

            t_verify_us += ggml_time_us() - t_verify_start;
            trace.add("target_verify_host", 2, n_verify + 1, t_verify_start);
            if (pending_draft.valid()) {
                const int64_t start = ggml_time_us();
                const int result = pending_draft.get();
                trace.add("join_draft", 1, n_verify + 1, start);
                if (result != 0) {
                    LOG_ERR("async draft decode failed\n");
                    return 1;
                }
            }
            ++n_verify;
            n_drafted += n_nodes;
            n_verify_nodes += batch_tgt.n_tokens;

            if (tree_diag) {
                LOG_INF("tree diag cost: round=%d draft-nodes=%d target-nodes=%d leaves=%d tree-build-us=%lld draft-decode-us=%lld target-verify-us=%lld\n",
                        n_verify, n_nodes, batch_tgt.n_tokens, layout.leaves,
                        (long long) t_round_tree_build_us, (long long) t_round_draft_decode_us,
                        (long long) (ggml_time_us() - t_verify_start));
            }

            LOG_DBG("tree verify call %d: draft nodes = %d, target batch nodes = %d, branches = %d\n",
                    n_verify, n_nodes, batch_tgt.n_tokens, n_seq_cur);
            ++n_past_tgt;
        }

        // the first token is always proposed by the target model before the speculation loop so we erase it here
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            drafts[s].tokens.erase(drafts[s].tokens.begin());
            drafts[s].dists.erase(drafts[s].dists.begin());
            drafts[s].diag_dists.erase(drafts[s].diag_dists.begin());
            drafts[s].ranks.erase(drafts[s].ranks.begin());
        }
    }

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("tree max depth  = %d\n", tree_max_depth);
    LOG_INF("tree max nodes  = %d\n", tree_max_nodes);
    LOG_INF("tree max leaves = %d\n", tree_max_leaves);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("n_verify  = %d\n", n_verify);
    LOG_INF("n_verify_nodes = %d\n", n_verify_nodes);
    LOG_INF("accept    = %.3f%%\n", n_drafted > 0 ? 100.0f * n_accept / n_drafted : 0.0f);
    LOG_INF("drafted per verify = %.3f\n", n_verify > 0 ? 1.0f * n_drafted / n_verify : 0.0f);
    LOG_INF("accepted per verify = %.3f\n", n_verify > 0 ? 1.0f * n_accept / n_verify : 0.0f);
    LOG_INF("target batch nodes per verify = %.3f\n", n_verify > 0 ? 1.0f * n_verify_nodes / n_verify : 0.0f);
    LOG_INF("tree nodes per verify = %.3f\n", n_verify > 0 ? 1.0f * n_tree_nodes / n_verify : 0.0f);
    LOG_INF("tree leaves per verify = %.3f\n", n_verify > 0 ? 1.0f * n_tree_leaves / n_verify : 0.0f);
    LOG_INF("tree depth per verify = %.3f\n", n_verify > 0 ? 1.0f * n_tree_depth / n_verify : 0.0f);
    LOG_INF("tree ancestor pairs per verify = %.3f\n", n_verify > 0 ? 1.0f * n_tree_ancestor_pairs / n_verify : 0.0f);
    LOG_INF("target padding nodes per verify = %.3f\n", n_verify > 0 ? 1.0f * n_verify_padding / n_verify : 0.0f);
    LOG_INF("tree layout cache builds = %lld\n", (long long) n_tree_layout_build);
    LOG_INF("tree layout cache hits = %lld\n", (long long) n_tree_layout_reuse);
    LOG_INF("tree draft decode calls = %lld\n", (long long) n_tree_draft_decode_calls);
    LOG_INF("tree draft nodes per call = %.3f\n", n_tree_draft_decode_calls > 0 ? 1.0f * n_tree_draft_decode_nodes / n_tree_draft_decode_calls : 0.0f);
    LOG_INF("tree pruned side candidates = %lld\n", (long long) n_tree_pruned_side_candidates);
    LOG_INF("tree ratio-pruned candidates = %lld\n", (long long) n_tree_pruned_ratio_candidates);
    LOG_INF("tree frontier candidates = %lld\n", (long long) n_tree_frontier_candidates);
    LOG_INF("tree frontier selected = %lld\n", (long long) n_tree_frontier_selected);
    LOG_INF("tree frontier pruned = %lld\n", (long long) n_tree_frontier_pruned);
    LOG_INF("tree frontier stopped leaves = %lld\n", (long long) n_tree_frontier_stopped_leaves);
    if (tree_frontier) {
        for (int depth = 0; depth < (int) frontier_calibration.buckets.size(); ++depth) {
            const auto & values = frontier_calibration.buckets[depth];
            LOG_INF("tree frontier calibration depth %d = r1 %.0f/%.0f, r2 %.0f/%.0f, r3 %.0f/%.0f, r4+ %.0f/%.0f\n",
                    depth + 1,
                    values[0].hits, values[0].trials,
                    values[1].hits, values[1].trials,
                    values[2].hits, values[2].trials,
                    values[3].hits, values[3].trials);
        }
    }
    LOG_INF("side branch entries = %lld\n", (long long) n_side_branch_entries);
    LOG_INF("side branch accepted tokens = %lld\n", (long long) n_side_branch_accepted);
    LOG_INF("side branch entry rate = %.3f%%\n", n_verify > 0 ? 100.0f * n_side_branch_entries / n_verify : 0.0f);
    LOG_INF("target rank histogram = top1:%lld top2:%lld top3:%lld miss:%lld\n",
            (long long) n_target_rank_1, (long long) n_target_rank_2,
            (long long) n_target_rank_3, (long long) n_target_rank_miss);
    const int64_t n_target_rank_total = n_target_rank_1 + n_target_rank_2 + n_target_rank_3 + n_target_rank_miss;
    LOG_INF("target coverage at 1/2/3 = %.3f%% / %.3f%% / %.3f%%\n",
            n_target_rank_total > 0 ? 100.0f * n_target_rank_1 / n_target_rank_total : 0.0f,
            n_target_rank_total > 0 ? 100.0f * (n_target_rank_1 + n_target_rank_2) / n_target_rank_total : 0.0f,
            n_target_rank_total > 0 ? 100.0f * (n_target_rank_1 + n_target_rank_2 + n_target_rank_3) / n_target_rank_total : 0.0f);
    LOG_INF("side accept length histogram = len1:%lld len2:%lld len3:%lld len4+:%lld\n",
            (long long) n_side_accept_len_1, (long long) n_side_accept_len_2,
            (long long) n_side_accept_len_3, (long long) n_side_accept_len_4_plus);
    LOG_INF("target tree verify time = %.3f ms\n", t_verify_us / 1000.0f);
    LOG_INF("target tree verify time per call = %.3f ms\n", n_verify > 0 ? t_verify_us / (1000.0f * n_verify) : 0.0f);
    LOG_INF("tree accept and sample time = %.3f ms\n", t_accept_us / 1000.0f);
    LOG_INF("target KV select time = %.3f ms\n", t_target_kv_select_us / 1000.0f);
    LOG_INF("draft sync time = %.3f ms\n", t_draft_sync_us / 1000.0f);
    LOG_INF("tree build time = %.3f ms\n", t_tree_build_us / 1000.0f);
    LOG_INF("draft sample time = %.3f ms\n", t_draft_sample_us / 1000.0f);
    LOG_INF("draft KV branch time = %.3f ms\n", t_draft_kv_branch_us / 1000.0f);
    LOG_INF("draft decode time = %.3f ms\n", t_draft_decode_us / 1000.0f);
    LOG_INF("tree DFS layout time = %.3f ms\n", t_tree_layout_us / 1000.0f);
    LOG_INF("target KV prepare time = %.3f ms\n", t_target_kv_prepare_us / 1000.0f);
    LOG_INF("target decode enqueue time = %.3f ms\n", t_target_enqueue_us / 1000.0f);
    LOG_INF("target decode sync time = %.3f ms\n", t_target_sync_us / 1000.0f);
    LOG_INF("draft decode enqueue time = %.3f ms\n", t_draft_enqueue_us / 1000.0f);
    LOG_INF("draft decode sync time = %.3f ms\n", t_draft_decode_sync_us / 1000.0f);

    const int64_t t_accounted_us = t_accept_us + t_target_kv_select_us + t_draft_sync_us +
        t_tree_build_us + t_tree_layout_us + t_target_kv_prepare_us + t_verify_us;
    const int64_t t_decode_us = t_dec_end - t_dec_start;
    LOG_INF("tree residual time = %.3f ms\n", std::max<int64_t>(0, t_decode_us - t_accounted_us) / 1000.0f);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    // TODO: print sampling/grammar timings for all drafts
    llama_perf_context_print(ctx_dft);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    for (int s = 0; s < n_seq_dft; ++s) {
        common_sampler_free(drafts[s].smpl);
    }

    llama_batch_free(batch_dft);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
