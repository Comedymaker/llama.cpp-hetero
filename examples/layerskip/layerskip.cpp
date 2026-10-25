#include "llama.h"
#include "../../src/llama-ext.h"
#include "../../src/llama-model.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <sched.h>
#include <pthread.h>
#endif

struct early_diag_sample {
    int64_t time_us = ggml_time_us();
    uint64_t read_bytes = 0;
    bool io_valid = false;
#if !defined(_WIN32)
    rusage usage{};
    bool usage_valid = getrusage(RUSAGE_SELF, &usage) == 0;
#endif
    early_diag_sample() {
#if defined(__linux__)
        std::ifstream input("/proc/self/io");
        std::string key;
        uint64_t value;
        while (input >> key >> value) {
            if (key == "read_bytes:") {
                read_bytes = value;
                io_valid = true;
                break;
            }
        }
#endif
    }
};

template<typename F>
static auto early_diag_call(const char * stage, F fn) -> decltype(fn()) {
    if (!std::getenv("LLAMA_EARLY_DIAG")) {
        return fn();
    }
    const early_diag_sample before;
    const auto result = fn();
    const early_diag_sample after;
#if !defined(_WIN32)
    const auto cpu_us = [](const rusage & usage) {
        return (int64_t) (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1000000 +
                usage.ru_utime.tv_usec + usage.ru_stime.tv_usec;
    };
    const bool usage_valid = before.usage_valid && after.usage_valid;
    std::fprintf(stderr, "early diag stage=%s wall_us=%lld cpu_us=%lld minor=%ld major=%ld voluntary=%ld involuntary=%ld usage_valid=%d read_bytes=%llu io_valid=%d\n",
            stage, (long long) (after.time_us - before.time_us),
            (long long) (cpu_us(after.usage) - cpu_us(before.usage)),
            after.usage.ru_minflt - before.usage.ru_minflt,
            after.usage.ru_majflt - before.usage.ru_majflt,
            after.usage.ru_nvcsw - before.usage.ru_nvcsw,
            after.usage.ru_nivcsw - before.usage.ru_nivcsw, usage_valid,
            (unsigned long long) (after.read_bytes - before.read_bytes), before.io_valid && after.io_valid);
#else
    std::fprintf(stderr, "early diag stage=%s wall_us=%lld usage_valid=0 io_valid=0\n",
            stage, (long long) (after.time_us - before.time_us));
#endif
    return result;
}

static void early_residency_diag(llama_model * model, int32_t exit_layer, const char * stage, int32_t cluster = -1) {
#if defined(__linux__)
    if (!std::getenv("LLAMA_EARLY_RESIDENCY_DIAG")) {
        return;
    }
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    uint64_t total = 0;
    uint64_t resident = 0;
    auto inspect = [&](const ggml_tensor * tensor) {
        if (!tensor || !tensor->data) {
            return;
        }
        const uintptr_t address = (uintptr_t) tensor->data;
        const uintptr_t base = address & ~(uintptr_t) (page - 1);
        const size_t count = (address - base + ggml_nbytes(tensor) + page - 1) / page;
        std::vector<unsigned char> pages(count);
        if (mincore((void *) base, count * page, pages.data()) != 0) {
            std::fprintf(stderr, "early residency stage=%s cluster=%d tensor=%s error=%d\n", stage, cluster, tensor->name, errno);
            return;
        }
        size_t present = 0;
        for (unsigned char value : pages) {
            present += value & 1;
        }
        total += count;
        resident += present;
        std::fprintf(stderr, "early residency stage=%s cluster=%d tensor=%s pages=%zu resident=%zu\n", stage, cluster, tensor->name, count, present);
    };
    inspect(model->tok_embd);
    inspect(model->output);
    inspect(model->output_norm);
    for (int32_t il = 0; il < exit_layer; ++il) {
        const auto & layer = model->layers.at(il);
        for (const ggml_tensor * tensor : {layer.attn_norm, layer.wq, layer.wk, layer.wv, layer.wo,
                layer.ffn_norm, layer.ffn_gate, layer.ffn_up, layer.ffn_down}) {
            inspect(tensor);
        }
    }
    std::fprintf(stderr, "early residency summary stage=%s cluster=%d pages=%llu resident=%llu\n", stage, cluster,
            (unsigned long long) total, (unsigned long long) resident);
#else
    GGML_UNUSED(model);
    GGML_UNUSED(exit_layer);
    GGML_UNUSED(stage);
    GGML_UNUSED(cluster);
#endif
}

struct options {
    std::string model;
    std::string prompt = "Explain speculative decoding briefly.";
    std::string prompt_file;
    bool prompt_given = false;
    int32_t n_predict = 32;
    int32_t n_ctx = 512;
    int32_t n_batch = 128;
    int32_t n_ubatch = 0;
    int32_t n_threads = 4;
    int32_t exit_layer = 8;
    int32_t draft_max = 6;
    bool draft_tree = false;
    int32_t tree_max_nodes = 0;
    int32_t tree_side_depth = 2;
    float tree_p_ratio = 0.30f;
    bool tree_single_branch = false;
    bool tree_gain_cost = false;
    int32_t tree_top_k = 2;
    double tree_verify_base_ms = 1000.0;
    double tree_node_ms = 20.0;
    double tree_leaf_ms = 5.0;
    bool tree_tail_prefetch_overlap = false;
    bool tree_tail_htp_overlap = false;
    bool ignore_eos = false;
    bool flash_attn_off = false;
    bool tail_head_on_device = false;
    bool target_only = false;
    bool split_target_only = false;
    bool tail_layerwise = false;
    bool tail_ffn_stream = false;
    bool tail_layer_stream = false;
    bool tail_stream_prefetch = false;
    bool tail_graph_reuse = false;
    int32_t tail_stream_cache_layers = 0;
    int32_t tail_stream_cluster_layers = 1;
    std::vector<int32_t> tail_stream_cluster_sizes;
    int32_t tail_stream_slots = 0;
    int32_t memory_budget_mib = 0;
    int32_t runtime_reserve_mib = 0;
    std::string logits_prefix;
    std::string tail_device;
    int32_t tail_offload_layers = 0;
};

static void print_usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s -m MODEL [-p PROMPT | -f PROMPT_FILE] [-n N] [-c N] [-b N] [-ub N] [-t N] "
            "--exit-layer N [--draft-max N] [--ignore-eos] [--flash-attn-off] "
            "[--draft-tree --tree-max-nodes N --tree-side-depth N --tree-p-ratio R --tree-single-branch "
            "--tree-tail-prefetch-overlap --tree-tail-htp-overlap] "
            "[--tree-gain-cost --tree-top-k N --tree-verify-base-ms N --tree-node-ms N --tree-leaf-ms N] "
            "[--target-only | --split-target-only] [--tail-device NAME --tail-offload-layers N] "
            "[--tail-layerwise | --tail-ffn-stream | --tail-layer-stream] "
            "[--tail-stream-prefetch] [--tail-graph-reuse] [--tail-stream-cache-layers N] [--tail-stream-cluster-layers N] "
            "[--tail-stream-cluster-sizes N,N,...] "
            "[--tail-stream-slots 1|2] "
            "[--memory-budget-mib N --runtime-reserve-mib N] "
            "[--logits-prefix PATH] "
            "[--tail-head-on-device (target-only diagnostic)]\n",
            argv0);
}

static bool parse_cluster_sizes(const char * value, std::vector<int32_t> & sizes) {
    int32_t size = 0;
    for (const char * p = value;; ++p) {
        if (*p >= '0' && *p <= '9') {
            const int32_t digit = *p - '0';
            if (size > (std::numeric_limits<int32_t>::max() - digit) / 10) {
                return false;
            }
            size = size * 10 + digit;
        } else if (*p == ',' || *p == '\0') {
            if (size == 0) {
                return false;
            }
            sizes.push_back(size);
            if (*p == '\0') {
                return true;
            }
            size = 0;
        } else {
            return false;
        }
    }
}

static bool parse_options(int argc, char ** argv, options & opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char * {
            return i + 1 < argc ? argv[++i] : nullptr;
        };

        const char * value = nullptr;
        if (arg == "-m" && (value = next())) {
            opts.model = value;
        } else if (arg == "-p" && (value = next())) {
            if (!opts.prompt_file.empty()) {
                return false;
            }
            opts.prompt = value;
            opts.prompt_given = true;
        } else if (arg == "-f" && (value = next())) {
            if (opts.prompt_given) {
                return false;
            }
            opts.prompt_file = value;
        } else if (arg == "-n" && (value = next())) {
            opts.n_predict = std::stoi(value);
        } else if (arg == "-c" && (value = next())) {
            opts.n_ctx = std::stoi(value);
        } else if (arg == "-b" && (value = next())) {
            opts.n_batch = std::stoi(value);
        } else if (arg == "-ub" && (value = next())) {
            opts.n_ubatch = std::stoi(value);
        } else if (arg == "-t" && (value = next())) {
            opts.n_threads = std::stoi(value);
        } else if (arg == "--exit-layer" && (value = next())) {
            opts.exit_layer = std::stoi(value);
        } else if (arg == "--draft-max" && (value = next())) {
            opts.draft_max = std::stoi(value);
        } else if (arg == "--draft-tree") {
            opts.draft_tree = true;
        } else if (arg == "--tree-max-nodes" && (value = next())) {
            opts.tree_max_nodes = std::stoi(value);
        } else if (arg == "--tree-side-depth" && (value = next())) {
            opts.tree_side_depth = std::stoi(value);
        } else if (arg == "--tree-p-ratio" && (value = next())) {
            opts.tree_p_ratio = std::stof(value);
        } else if (arg == "--tree-gain-cost") {
            opts.tree_gain_cost = true;
            opts.draft_tree = true;
        } else if (arg == "--tree-top-k" && (value = next())) {
            opts.tree_top_k = std::stoi(value);
        } else if (arg == "--tree-verify-base-ms" && (value = next())) {
            opts.tree_verify_base_ms = std::stod(value);
        } else if (arg == "--tree-node-ms" && (value = next())) {
            opts.tree_node_ms = std::stod(value);
        } else if (arg == "--tree-leaf-ms" && (value = next())) {
            opts.tree_leaf_ms = std::stod(value);
        } else if (arg == "--tree-single-branch") {
            opts.tree_single_branch = true;
        } else if (arg == "--tree-tail-prefetch-overlap") {
            opts.tree_tail_prefetch_overlap = true;
        } else if (arg == "--tree-tail-htp-overlap") {
            opts.tree_tail_htp_overlap = true;
        } else if (arg == "--ignore-eos") {
            opts.ignore_eos = true;
        } else if (arg == "--flash-attn-off") {
            opts.flash_attn_off = true;
        } else if (arg == "--tail-head-on-device") {
            opts.tail_head_on_device = true;
        } else if (arg == "--target-only") {
            opts.target_only = true;
        } else if (arg == "--split-target-only") {
            opts.split_target_only = true;
        } else if (arg == "--tail-ffn-stream") {
            opts.tail_ffn_stream = true;
        } else if (arg == "--tail-layer-stream") {
            opts.tail_layer_stream = true;
        } else if (arg == "--tail-stream-prefetch" || arg == "--tail-ffn-prefetch") {
            opts.tail_stream_prefetch = true;
        } else if (arg == "--tail-graph-reuse") {
            opts.tail_graph_reuse = true;
        } else if ((arg == "--tail-stream-cache-layers" || arg == "--tail-ffn-cache-layers") && (value = next())) {
            opts.tail_stream_cache_layers = std::stoi(value);
        } else if (arg == "--tail-stream-cluster-layers" && (value = next())) {
            opts.tail_stream_cluster_layers = std::stoi(value);
        } else if (arg == "--tail-stream-cluster-sizes" && (value = next())) {
            if (!opts.tail_stream_cluster_sizes.empty() || !parse_cluster_sizes(value, opts.tail_stream_cluster_sizes)) {
                return false;
            }
        } else if (arg == "--tail-stream-slots" && (value = next())) {
            opts.tail_stream_slots = std::stoi(value);
        } else if (arg == "--memory-budget-mib" && (value = next())) {
            opts.memory_budget_mib = std::stoi(value);
        } else if (arg == "--runtime-reserve-mib" && (value = next())) {
            opts.runtime_reserve_mib = std::stoi(value);
        } else if (arg == "--tail-layerwise") {
            opts.tail_layerwise = true;
        } else if (arg == "--logits-prefix" && (value = next())) {
            opts.logits_prefix = value;
        } else if (arg == "--tail-device" && (value = next())) {
            opts.tail_device = value;
        } else if (arg == "--tail-offload-layers" && (value = next())) {
            opts.tail_offload_layers = std::stoi(value);
        } else {
            return false;
        }
    }

    if (opts.draft_tree && opts.tree_max_nodes == 0) {
        opts.tree_max_nodes = opts.draft_max * (opts.draft_max + 3) / 2;
    }
#if defined(LLAMA_LAYERSKIP_OLD_DIRECT_BACKENDS)
    if (opts.tail_stream_slots == 2 || opts.tree_tail_htp_overlap) {
        std::fprintf(stderr, "old direct backend build supports one slot and no HTP overlap\n");
        return false;
    }
#endif

    return !opts.model.empty() && !(opts.target_only && opts.split_target_only) && opts.n_predict > 0 && opts.n_ctx > 0 &&
            opts.n_batch >= opts.draft_max &&
            (opts.target_only || opts.split_target_only || opts.n_batch > opts.draft_max) &&
            (!opts.draft_tree || (opts.tree_max_nodes > opts.draft_max && opts.n_batch > opts.tree_max_nodes &&
                    opts.tree_side_depth > 0 && opts.tree_p_ratio >= 0.0f && opts.tree_p_ratio <= 1.0f)) &&
            (!opts.tree_gain_cost || (opts.tree_max_nodes <= 63 && opts.tree_top_k >= 2 && opts.tree_top_k <= 8 &&
                    !opts.tree_single_branch && !opts.tree_tail_prefetch_overlap && !opts.tree_tail_htp_overlap &&
                    !opts.target_only && !opts.split_target_only &&
                    std::isfinite(opts.tree_verify_base_ms) && opts.tree_verify_base_ms > 0 &&
                    std::isfinite(opts.tree_node_ms) && opts.tree_node_ms > 0 &&
                    std::isfinite(opts.tree_leaf_ms) && opts.tree_leaf_ms >= 0)) &&
            (!opts.tree_tail_prefetch_overlap || (opts.draft_tree && opts.tail_stream_prefetch)) &&
            (!opts.tree_tail_htp_overlap || (opts.draft_tree && opts.tail_stream_prefetch && opts.tail_layer_stream)) &&
            opts.n_ubatch >= 0 && opts.n_threads > 0 && opts.exit_layer > 0 && opts.draft_max > 0 &&
            (opts.tail_device.empty() == (opts.tail_offload_layers == 0)) &&
            (!(opts.tail_ffn_stream && opts.tail_layer_stream)) &&
            (!(opts.tail_ffn_stream || opts.tail_layer_stream) || !opts.tail_device.empty()) &&
            (!opts.tail_stream_prefetch || opts.tail_ffn_stream || opts.tail_layer_stream) &&
            (!opts.tail_graph_reuse || opts.tail_ffn_stream || opts.tail_layer_stream) &&
            (opts.tail_stream_cache_layers >= 0) &&
            (opts.tail_stream_cluster_layers > 0) &&
            (opts.tail_stream_slots >= 0 && opts.tail_stream_slots <= 2) &&
            (opts.tail_stream_slots == 0 || opts.tail_layer_stream) &&
            (opts.tail_stream_cluster_layers == 1 || opts.tail_layer_stream) &&
            (opts.tail_stream_cluster_sizes.empty() || opts.tail_layer_stream) &&
            (opts.memory_budget_mib >= 0) &&
            (opts.runtime_reserve_mib >= 0) &&
            (opts.memory_budget_mib == 0 || (opts.tail_layer_stream && !opts.target_only &&
                    opts.runtime_reserve_mib > 0)) &&
            (opts.tail_stream_cache_layers == 0 || opts.tail_ffn_stream || opts.tail_layer_stream) &&
            (!opts.tail_layerwise || !opts.tail_device.empty()) &&
            (!opts.tail_head_on_device || (opts.target_only && !opts.tail_device.empty())) &&
            (opts.logits_prefix.empty() || opts.target_only || opts.split_target_only || opts.tree_gain_cost);
}

static void batch_add_token(llama_batch & batch, llama_token token, llama_pos pos, bool output) {
    const int32_t i = batch.n_tokens++;
    batch.token[i] = token;
    batch.pos[i] = pos;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = output;
}

static void batch_add_embd(llama_batch & batch, const float * embd, int32_t n_embd, llama_pos pos, bool output) {
    const int32_t i = batch.n_tokens++;
    std::copy(embd, embd + n_embd, batch.embd + (size_t) i * n_embd);
    batch.pos[i] = pos;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = output;
}

static void batch_add_token_seqs(llama_batch & batch, llama_token token, llama_pos pos,
        llama_seq_id seq0, llama_seq_id seq1, bool output) {
    const int32_t i = batch.n_tokens++;
    batch.token[i] = token;
    batch.pos[i] = pos;
    batch.n_seq_id[i] = seq1 < 0 ? 1 : 2;
    batch.seq_id[i][0] = seq0;
    if (seq1 >= 0) {
        batch.seq_id[i][1] = seq1;
    }
    batch.logits[i] = output;
}

static void batch_add_embd_seqs(llama_batch & batch, const float * embd, int32_t n_embd, llama_pos pos,
        llama_seq_id seq0, llama_seq_id seq1, bool output) {
    const int32_t i = batch.n_tokens++;
    std::copy(embd, embd + n_embd, batch.embd + (size_t) i * n_embd);
    batch.pos[i] = pos;
    batch.n_seq_id[i] = seq1 < 0 ? 1 : 2;
    batch.seq_id[i][0] = seq0;
    if (seq1 >= 0) {
        batch.seq_id[i][1] = seq1;
    }
    batch.logits[i] = output;
}

static void batch_add_embd_seq_list(llama_batch & batch, const float * embd, int32_t n_embd, llama_pos pos,
        const std::vector<llama_seq_id> & seq_ids, bool output) {
    const int32_t i = batch.n_tokens++;
    std::copy(embd, embd + n_embd, batch.embd + (size_t) i * n_embd);
    batch.pos[i] = pos;
    batch.n_seq_id[i] = (int32_t) seq_ids.size();
    std::copy(seq_ids.begin(), seq_ids.end(), batch.seq_id[i]);
    batch.logits[i] = output;
}

struct greedy_pair {
    llama_token first = LLAMA_TOKEN_NULL;
    llama_token second = LLAMA_TOKEN_NULL;
    float ratio = 0.0f;
};

static std::vector<uint8_t> eog_mask;

static void mask_eog_logits(std::vector<float> & logits) {
    for (size_t i = 0; i < eog_mask.size(); ++i) {
        if (eog_mask[i]) { logits[i] = -INFINITY; }
    }
}

static greedy_pair top2_tokens(const float * logits, int32_t n_vocab) {
    greedy_pair result;
    float first = -INFINITY;
    float second = -INFINITY;
    for (int32_t token = 0; token < n_vocab; ++token) {
        if (!eog_mask.empty() && eog_mask[token]) { continue; }
        const float value = logits[token];
        if (value > first) {
            second = first;
            result.second = result.first;
            first = value;
            result.first = token;
        } else if (value > second) {
            second = value;
            result.second = token;
        }
    }
    if (result.second != LLAMA_TOKEN_NULL) {
        result.ratio = std::exp(second - first);
    }
    return result;
}

static llama_token argmax_token(const float * logits, int32_t n_vocab) {
    if (!eog_mask.empty()) {
        llama_token best = LLAMA_TOKEN_NULL;
        for (int32_t token = 0; token < n_vocab; ++token) {
            if (!eog_mask[token] && (best == LLAMA_TOKEN_NULL || logits[token] > logits[best])) { best = token; }
        }
        return best;
    }
    return (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
}

static bool copy_logits(llama_context * ctx, int32_t row, int32_t n_vocab, std::vector<float> & dst) {
    float * logits = llama_get_logits_ith(ctx, row);
    if (!logits) {
        return false;
    }
    dst.assign(logits, logits + n_vocab);
    mask_eog_logits(dst);
    return true;
}

static std::string token_to_tsv(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buf(128);
    int32_t n = llama_token_to_piece(vocab, token, buf.data(), (int32_t) buf.size(), 0, true);
    if (n < 0) {
        buf.resize(-n);
        n = llama_token_to_piece(vocab, token, buf.data(), (int32_t) buf.size(), 0, true);
    }
    std::string result;
    for (int32_t i = 0; i < n; ++i) {
        switch (buf[i]) {
            case '\\': result += "\\\\"; break;
            case '\t': result += "\\t"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            default:   result += buf[i]; break;
        }
    }
    return result;
}

struct logits_dump {
    std::ofstream f32;
    std::ofstream tsv;

    bool open(const std::string & prefix, int32_t n_vocab) {
        f32.open(prefix + ".f32", std::ios::binary);
        tsv.open(prefix + ".tsv");
        if (!f32 || !tsv) {
            return false;
        }
        tsv << "# n_vocab\t" << n_vocab << '\n';
        tsv << "# position=chosen_token_position\n";
        tsv << "step\tposition\ttop1_token\ttop1_id\ttop1_logit\ttop2_id\ttop2_logit\tmargin\n";
        tsv.flush();
        return tsv.good();
    }

    bool write(const llama_vocab * vocab, const float * logits, int32_t n_vocab, int32_t step, llama_pos pos) {
        f32.write(reinterpret_cast<const char *>(logits), (size_t) n_vocab * sizeof(*logits));
        const llama_token top1 = argmax_token(logits, n_vocab);
        const llama_token top2 = top1 == 0 ? 1 : 0;
        llama_token next = top2;
        for (int32_t i = 0; i < n_vocab; ++i) {
            if (i != top1 && logits[i] > logits[next]) {
                next = i;
            }
        }
        tsv << std::setprecision(9) << step << '\t' << pos << '\t' << token_to_tsv(vocab, top1) << '\t' << top1 << '\t'
                << logits[top1] << '\t' << next << '\t' << logits[next] << '\t' << logits[top1] - logits[next] << '\n';
        f32.flush();
        tsv.flush();
        return f32.good() && tsv.good();
    }
};

static void print_token(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buf(128);
    int32_t n = llama_token_to_piece(vocab, token, buf.data(), (int32_t) buf.size(), 0, true);
    if (n < 0) {
        buf.resize(-n);
        n = llama_token_to_piece(vocab, token, buf.data(), (int32_t) buf.size(), 0, true);
    }
    if (n > 0) {
        std::fwrite(buf.data(), 1, n, stdout);
        std::fflush(stdout);
    }
}

static void print_token_trace(const std::vector<llama_token> & tokens) {
    std::fprintf(stderr, "generated token ids =");
    for (size_t i = 0; i < std::min<size_t>(tokens.size(), 32); ++i) {
        std::fprintf(stderr, " %d", tokens[i]);
    }
    std::fprintf(stderr, "\n");
}

struct tail_decoder {
    struct prefetch_result {
        uint64_t bytes = 0;
        uint64_t time_us = 0;
    };

    struct htp_preload_result {
        int64_t bytes = -1;
        uint64_t time_us = 0;
        int64_t start_us = 0;
        int64_t end_us = 0;
        int affinity_error = 0;
    };

    llama_model * model;
    int32_t il_start;
    int32_t il_end;
    int32_t n_embd;
    int32_t n_vocab;
    bool layerwise;
    bool stream_ffn;
    bool stream_layer;
    bool prefetch_stream;
    bool graph_reuse;
    int32_t cache_layers;
    bool fixed_htp_cache;
    int32_t cluster_layers;
    std::vector<int32_t> stream_cluster_sizes;
    int32_t stream_slots;
    bool trace_clusters = std::getenv("LLAMA_CLUSTER_TRACE") != nullptr;
    bool trace_generation = false;
    uint64_t trace_call = 0;
    llama_context_params params;
    std::vector<llama_context *> contexts;
    std::vector<int32_t> context_starts;
    std::vector<int32_t> context_ends;
    llama_batch scratch = {};
    llama_batch tree_htp_batch = {};
    std::vector<float> hidden;
    std::vector<std::vector<float>> logits;
    uint64_t layer_loads = 0;
    uint64_t stream_bytes = 0;
    uint64_t prefetch_calls = 0;
    uint64_t prefetch_failures = 0;
    uint64_t discard_failures = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_bytes = 0;
    uint64_t prefetch_time_us = 0;
    uint64_t prefetch_wait_us = 0;
    uint64_t tree_prefetch_calls = 0;
    uint64_t tree_prefetch_time_us = 0;
    uint64_t tree_prefetch_wait_us = 0;
    uint64_t tree_htp_preload_calls = 0;
    uint64_t tree_htp_preload_failures = 0;
    uint64_t tree_htp_preload_time_us = 0;
    uint64_t tree_htp_preload_wait_us = 0;
    uint64_t overlap_htp_preload_calls = 0;
    uint64_t overlap_htp_preload_failures = 0;
    uint64_t overlap_htp_preload_time_us = 0;
    uint64_t overlap_htp_preload_wait_us = 0;
    uint64_t htp_prepare_time_us = 0;
    uint64_t htp_io_affinity_failures = 0;
    int htp_io_cpu = -1;
    uint64_t htp_preload_calls = 0;
    uint64_t htp_preload_failures = 0;
    uint64_t htp_preload_bytes = 0;
    uint64_t htp_preload_time_us = 0;
    uint64_t htp_compute_wait_us = 0;
    uint64_t htp_direct_loads = 0;
    uint64_t htp_direct_bytes = 0;
    std::future<prefetch_result> prefetch;
    int32_t prefetch_index = -1;
    bool tree_prefetch_active = false;
    std::future<htp_preload_result> tree_htp_preload;
    int32_t tree_htp_preload_index = -1;
    bool htp_preload_is_tree = false;
    std::vector<uint8_t> htp_resident;

    tail_decoder(llama_model * model, int32_t il_start, int32_t il_end, int32_t n_embd, int32_t n_vocab,
            llama_context_params params, bool layerwise, bool stream_ffn, bool stream_layer,
            bool prefetch_stream, bool graph_reuse, int32_t cache_layers, bool fixed_htp_cache,
            int32_t cluster_layers, const std::vector<int32_t> & stream_cluster_sizes, int32_t stream_slots) :
            model(model), il_start(il_start), il_end(il_end), n_embd(n_embd), n_vocab(n_vocab), layerwise(layerwise),
            stream_ffn(stream_ffn), stream_layer(stream_layer), prefetch_stream(prefetch_stream), graph_reuse(graph_reuse),
            cache_layers(cache_layers), fixed_htp_cache(fixed_htp_cache), cluster_layers(cluster_layers),
            stream_cluster_sizes(stream_cluster_sizes),
            stream_slots(stream_slots), params(params) {
    }

    ~tail_decoder() {
        stop_prefetch();
        clear();
    }

    void stop_prefetch() {
        if (prefetch.valid()) {
            prefetch.wait();
        }
    }

    void clear() {
        stop_prefetch();
        if (tree_htp_preload.valid()) {
            tree_htp_preload.wait();
        }
        if (scratch.token || scratch.embd) {
            llama_batch_free(scratch);
            scratch = {};
        }
        if (tree_htp_batch.token || tree_htp_batch.embd) {
            llama_batch_free(tree_htp_batch);
            tree_htp_batch = {};
        }
        for (auto * ctx : contexts) {
            llama_free(ctx);
        }
        contexts.clear();
        context_starts.clear();
        context_ends.clear();
    }

    bool init() {
        if (const char * value = std::getenv("LLAMA_HTP_IO_CPU")) {
#if defined(__linux__)
            char * end = nullptr;
            errno = 0;
            const long cpu = std::strtol(value, &end, 10);
            if (errno || end == value || *end || cpu < 0 || cpu >= CPU_SETSIZE) {
                std::fprintf(stderr, "invalid LLAMA_HTP_IO_CPU: %s\n", value);
                return false;
            }
            htp_io_cpu = (int) cpu;
#else
            std::fprintf(stderr, "LLAMA_HTP_IO_CPU requires Linux\n");
            return false;
#endif
        }
        if (!layerwise) {
            contexts.push_back(llama_init_from_model(model, params));
            return contexts.back() != nullptr;
        }

        scratch = llama_batch_init(params.n_batch, n_embd, params.n_seq_max);
        tree_htp_batch = llama_batch_init(params.n_batch, n_embd, params.n_seq_max);
        size_t stream_cluster_index = 0;
        for (int32_t il = il_start; il < il_end;) {
            const bool cached = il < il_start + cache_layers;
            const int32_t range_end = cached ? il_start + cache_layers : il_end;
            const int32_t next = !cached && !stream_cluster_sizes.empty() ?
                    il + stream_cluster_sizes[stream_cluster_index++] : std::min(range_end, il + cluster_layers);
            auto layer_params = params;
            layer_params.n_layer_start = il;
            layer_params.n_layer_end = next;
            layer_params.stream_ffn = stream_ffn;
            layer_params.stream_layer = stream_layer;
            llama_context * ctx = llama_init_from_model(model, layer_params);
            if (!ctx) {
                return false;
            }
            if (next < il_end) {
                llama_set_embeddings_layer_inp(ctx, next, true);
            }
            llama_release_compute(ctx);
            contexts.push_back(ctx);
            context_starts.push_back(il);
            context_ends.push_back(next);
            il = next;
        }
        htp_resident.assign(contexts.size(), 0);
        if (fixed_htp_cache) {
            for (int32_t index = 0; index < (int32_t) contexts.size(); ++index) {
                htp_resident[index] = is_context_cached(index);
            }
        }
        for (int32_t il = il_start; il < il_start + cache_layers; ++il) {
            cache_bytes += stream_layer_bytes(il);
        }
        for (int32_t il = il_start + cache_layers; il < il_end; ++il) {
            discard_stream_pages(il);
        }
        return true;
    }

    std::vector<const ggml_tensor *> stream_tensors(int32_t il) const {
        const auto & layer = model->layers.at(il);
        std::vector<const ggml_tensor *> tensors = {
            layer.ffn_gate, layer.ffn_up, layer.ffn_down,
            layer.ffn_gate_b, layer.ffn_up_b, layer.ffn_down_b,
        };
        if (stream_layer) {
            tensors.insert(tensors.end(), {
                layer.attn_norm, layer.attn_norm_b,
                layer.wq, layer.wk, layer.wv, layer.wo,
                layer.wq_b, layer.wk_b, layer.wv_b, layer.wo_b,
                layer.wqkv, layer.wqkv_b,
                layer.ffn_norm, layer.ffn_norm_b,
            });
        }
        return tensors;
    }

    uint64_t advise_stream_pages(int32_t il, bool will_need) {
        uint64_t bytes = 0;
        for (const ggml_tensor * tensor : stream_tensors(il)) {
            bytes += model->mmap_advise(tensor, will_need);
        }
        return bytes;
    }

    void discard_stream_pages(int32_t il) {
        if (advise_stream_pages(il, false) == 0) {
            ++discard_failures;
        }
    }

    uint64_t load_stream_context(int32_t index) {
        ++prefetch_calls;
        const uint64_t bytes = advise_stream_context(index, true);
        if (bytes == 0) {
            ++prefetch_failures;
        }
        return bytes;
    }

    void start_prefetch(int32_t index) {
        if (is_context_cached(index) && htp_resident[index]) {
            return;
        }
        GGML_ASSERT(!prefetch.valid());
        prefetch_index = index;
        try {
            prefetch = std::async(std::launch::async, [this, index]() {
                const auto start = std::chrono::steady_clock::now();
                const uint64_t bytes = advise_stream_context(index, true);
                const auto end = std::chrono::steady_clock::now();
                return prefetch_result { bytes, (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() };
            });
            ++prefetch_calls;
        } catch (...) {
            prefetch_index = -1;
            ++prefetch_failures;
        }
    }

    void start_tree_prefetch() {
        if (!layerwise || !prefetch_stream || !(stream_ffn || stream_layer) || prefetch.valid()) {
            return;
        }
        for (int32_t index = 0; index < (int32_t) contexts.size(); ++index) {
            if (!is_context_cached(index) || !htp_resident[index]) {
                start_prefetch(index);
                if (prefetch.valid()) {
                    tree_prefetch_active = true;
                    ++tree_prefetch_calls;
                }
                return;
            }
        }
    }

    void finish_prefetch(int32_t index) {
        if (is_context_cached(index) && htp_resident[index]) {
            return;
        }
        if (prefetch.valid() && prefetch_index == index) {
            const auto start = std::chrono::steady_clock::now();
            try {
                const prefetch_result result = prefetch.get();
                const auto end = std::chrono::steady_clock::now();
                stream_bytes += result.bytes;
                prefetch_time_us += result.time_us;
                const uint64_t wait_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                prefetch_wait_us += wait_us;
                if (tree_prefetch_active) {
                    tree_prefetch_time_us += result.time_us;
                    tree_prefetch_wait_us += wait_us;
                    tree_prefetch_active = false;
                }
                prefetch_index = -1;
                if (result.bytes == 0) {
                    ++prefetch_failures;
                }
                return;
            } catch (...) {
                prefetch_index = -1;
                tree_prefetch_active = false;
                ++prefetch_failures;
            }
        }
        stream_bytes += load_stream_context(index);
    }

    void finish_tree_prefetch() {
        if (prefetch.valid()) {
            finish_prefetch(prefetch_index);
        }
    }

    bool tree_prefetch_ready() const {
        return !prefetch.valid() ||
                prefetch.wait_for(std::chrono::microseconds(0)) == std::future_status::ready;
    }

    bool is_stream_cached(int32_t il) const {
        return il < il_start + cache_layers;
    }

    bool is_context_cached(int32_t index) const {
        return context_ends[index] <= il_start + cache_layers;
    }

    uint64_t stream_context_bytes(int32_t index) const {
        uint64_t bytes = 0;
        for (int32_t il = context_starts[index]; il < context_ends[index]; ++il) {
            bytes += stream_layer_bytes(il);
        }
        return bytes;
    }

    uint64_t advise_stream_context(int32_t index, bool will_need) {
        uint64_t bytes = 0;
        for (int32_t il = context_starts[index]; il < context_ends[index]; ++il) {
            bytes += advise_stream_pages(il, will_need);
        }
        return bytes;
    }

    void discard_stream_context(int32_t index) {
        early_residency_diag(model, il_start, "discard.before", index);
        if (advise_stream_context(index, false) == 0) {
            ++discard_failures;
        }
        early_residency_diag(model, il_start, "discard.after", index);
    }

    uint64_t stream_layer_bytes(int32_t il) const {
        uint64_t bytes = 0;
        for (const ggml_tensor * tensor : stream_tensors(il)) {
            if (tensor) {
                bytes += ggml_nbytes(tensor);
            }
        }
        return bytes;
    }

    void trace_cluster(int32_t index, const char * event, int64_t start, int64_t end, int64_t bytes = 0) const {
        if (trace_clusters) {
            std::fprintf(stderr, "cluster trace phase=%s call=%llu index=%d first=%d last=%d cached=%d event=%s start_us=%lld end_us=%lld bytes=%lld\n",
                    trace_generation ? "generation" : "prompt", (unsigned long long) trace_call, index,
                    context_starts[index], context_ends[index], is_context_cached(index), event,
                    (long long) start, (long long) end, (long long) bytes);
        }
    }

    bool preload_htp_weights(int32_t index, const llama_batch & batch, bool direct) {
        const int64_t trace_start = trace_clusters ? ggml_time_us() : 0;
        const auto start = std::chrono::steady_clock::now();
        const int64_t bytes = direct ? llama_preload_compute_weights_direct(contexts[index], batch) :
                llama_preload_compute_weights(contexts[index], batch);
        const auto end = std::chrono::steady_clock::now();
        trace_cluster(index, "preload_sync", trace_start, trace_clusters ? ggml_time_us() : 0, bytes);
        ++htp_preload_calls;
        htp_preload_time_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        if (bytes < 0) {
            ++htp_preload_failures;
            return false;
        }
        htp_preload_bytes += (uint64_t) bytes;
        if (direct) {
            ++htp_direct_loads;
            htp_direct_bytes += (uint64_t) bytes;
        }
        return true;
    }

    bool start_htp_preload(int32_t index, const llama_batch & input, bool is_tree) {
#if defined(LLAMA_LAYERSKIP_OLD_DIRECT_BACKENDS)
        GGML_UNUSED(index);
        GGML_UNUSED(input);
        GGML_UNUSED(is_tree);
        return false;
#else
        if (!layerwise || !prefetch_stream || !(stream_ffn || stream_layer) ||
                index < 0 || index >= (int32_t) contexts.size() ||
                input.n_tokens > (int32_t) params.n_ubatch || tree_htp_preload.valid()) {
            return false;
        }

        tree_htp_batch.n_tokens = 0;
        for (int32_t i = 0; i < input.n_tokens; ++i) {
            const int32_t j = tree_htp_batch.n_tokens++;
            std::copy(input.embd + (size_t) i * n_embd, input.embd + (size_t) (i + 1) * n_embd,
                    tree_htp_batch.embd + (size_t) j * n_embd);
            tree_htp_batch.pos[j] = input.pos[i];
            tree_htp_batch.n_seq_id[j] = input.n_seq_id[i];
            for (int32_t s = 0; s < input.n_seq_id[i]; ++s) {
                tree_htp_batch.seq_id[j][s] = input.seq_id[i][s];
            }
            tree_htp_batch.logits[j] = input.logits[i];
        }

        tree_htp_preload_index = index;
        htp_preload_is_tree = is_tree;
        const int64_t prepare_start = ggml_time_us();
        if (!llama_prepare_compute_weights_direct(contexts[index], tree_htp_batch)) {
            tree_htp_preload_index = -1;
            return false;
        }
        const int64_t prepare_end = ggml_time_us();
        htp_prepare_time_us += prepare_end - prepare_start;
        trace_cluster(index, "prepare_main", prepare_start, prepare_end);
        try {
            tree_htp_preload = std::async(std::launch::async, [this, index]() {
#if defined(__linux__)
                pthread_setname_np(pthread_self(), "llama-htp-io");
                if (htp_io_cpu >= 0) {
                    cpu_set_t mask;
                    CPU_ZERO(&mask);
                    CPU_SET(htp_io_cpu, &mask);
                    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
                        htp_preload_result result;
                        result.affinity_error = errno;
                        return result;
                    }
                }
#endif
                const int64_t trace_start = trace_clusters ? ggml_time_us() : 0;
                const auto start = std::chrono::steady_clock::now();
                const int64_t bytes = llama_read_prepared_compute_weights(contexts[index]);
                const auto end = std::chrono::steady_clock::now();
                return htp_preload_result {
                    bytes,
                    (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(),
                    trace_start,
                    trace_clusters ? ggml_time_us() : 0,
                };
            });
            if (is_tree) {
                ++tree_htp_preload_calls;
            } else {
                ++overlap_htp_preload_calls;
            }
            return true;
        } catch (...) {
            llama_commit_prepared_compute_weights(contexts[index]);
            tree_htp_preload_index = -1;
            if (is_tree) {
                ++tree_htp_preload_failures;
            } else {
                ++overlap_htp_preload_failures;
            }
            return false;
        }
#endif
    }

    bool start_tree_htp_preload(const llama_batch & input) {
        int32_t index = -1;
        for (int32_t i = 0; i < (int32_t) contexts.size(); ++i) {
            if (!htp_resident[i]) {
                index = i;
                break;
            }
        }
        return start_htp_preload(index, input, true);
    }

    bool finish_tree_htp_preload(int32_t index) {
#if defined(LLAMA_LAYERSKIP_OLD_DIRECT_BACKENDS)
        GGML_UNUSED(index);
        return false;
#else
        if (!tree_htp_preload.valid() || tree_htp_preload_index != index) {
            return false;
        }

        const auto start = std::chrono::steady_clock::now();
        const int64_t trace_wait_start = trace_clusters ? ggml_time_us() : 0;
        try {
            const htp_preload_result result = tree_htp_preload.get();
            const bool committed = llama_commit_prepared_compute_weights(contexts[index]);
            if (result.affinity_error) {
                ++htp_io_affinity_failures;
                std::fprintf(stderr, "HTP I/O affinity failed: cpu=%d errno=%d\n", htp_io_cpu, result.affinity_error);
            }
            trace_cluster(index, "preload_async", result.start_us, result.end_us, result.bytes);
            trace_cluster(index, "preload_wait", trace_wait_start, trace_clusters ? ggml_time_us() : 0);
            const auto end = std::chrono::steady_clock::now();
            const uint64_t wait_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            tree_htp_preload_index = -1;
            if (htp_preload_is_tree) {
                tree_htp_preload_time_us += result.time_us;
                tree_htp_preload_wait_us += wait_us;
            } else {
                overlap_htp_preload_time_us += result.time_us;
                overlap_htp_preload_wait_us += wait_us;
            }
            ++htp_preload_calls;
            htp_preload_time_us += result.time_us;
            if (result.bytes < 0 || !committed) {
                if (htp_preload_is_tree) {
                    ++tree_htp_preload_failures;
                } else {
                    ++overlap_htp_preload_failures;
                }
                ++htp_preload_failures;
                return false;
            }
            htp_preload_bytes += (uint64_t) result.bytes;
            ++htp_direct_loads;
            htp_direct_bytes += (uint64_t) result.bytes;
            return true;
        } catch (...) {
            llama_commit_prepared_compute_weights(contexts[index]);
            tree_htp_preload_index = -1;
            if (htp_preload_is_tree) {
                ++tree_htp_preload_failures;
            } else {
                ++overlap_htp_preload_failures;
            }
            ++htp_preload_failures;
            return false;
        }
#endif
    }

    bool capture_logits(llama_context * ctx, const llama_batch & batch) {
        logits.clear();
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i]) {
                continue;
            }
            float * src = llama_get_logits_ith(ctx, i);
            if (!src) {
                return false;
            }
            if (std::any_of(src, src + n_vocab, [](float value) { return !std::isfinite(value); })) {
                std::fprintf(stderr, "Tail logits contain non-finite values at row %d\n", i);
                return false;
            }
            logits.emplace_back(src, src + n_vocab);
        }
        return !logits.empty();
    }

    bool decode(const llama_batch & input) {
        early_residency_diag(model, il_start, "tail.enter");
        ++trace_call;
        if (!layerwise) {
            return llama_decode(contexts[0], input) == 0 && capture_logits(contexts[0], input);
        }

        std::vector<int8_t> final_output(input.logits, input.logits + input.n_tokens);
        scratch.n_tokens = 0;
        for (int32_t i = 0; i < input.n_tokens; ++i) {
            const int32_t j = scratch.n_tokens++;
            std::copy(input.embd + (size_t) i * n_embd, input.embd + (size_t) (i + 1) * n_embd,
                    scratch.embd + (size_t) j * n_embd);
            scratch.pos[j] = input.pos[i];
            scratch.n_seq_id[j] = input.n_seq_id[i];
            for (int32_t s = 0; s < input.n_seq_id[i]; ++s) {
                scratch.seq_id[j][s] = input.seq_id[i][s];
            }
            scratch.logits[j] = contexts.size() == 1 ? final_output[i] : true;
        }

        const llama_batch * current = &scratch;
        const bool streaming = stream_ffn || stream_layer;
        const bool direct_mode = streaming && prefetch_stream && input.n_tokens <= (int32_t) params.n_ubatch;
        const bool overlap_htp_preload = direct_mode && stream_slots > 1;
        if (streaming && prefetch.valid() && prefetch_index == 0) {
            finish_prefetch(0);
        } else if (streaming && (!prefetch_stream || !direct_mode) && !htp_resident[0]) {
            stream_bytes += load_stream_context(0);
        }
        if (direct_mode && !htp_resident[0]) {
            if (!finish_tree_htp_preload(0)) {
                preload_htp_weights(0, scratch, true);
            }
        }
        for (int32_t index = 0; index < (int32_t) contexts.size(); ++index) {
            llama_context * ctx = contexts[index];
            const int32_t il_begin = context_starts[index];
            const int32_t il_end = context_ends[index];
            early_residency_diag(model, il_start, "cluster.before_load", index);
            if (direct_mode && !overlap_htp_preload && index > 0 && !htp_resident[index]) {
                if (prefetch.valid() && prefetch_index == index) {
                    finish_prefetch(index);
                }
                if (!finish_tree_htp_preload(index)) {
                    preload_htp_weights(index, scratch, true);
                }
            }
            if (streaming && prefetch_stream && !direct_mode && index + 1 < (int32_t) contexts.size() &&
                    !htp_resident[index + 1]) {
                if (!prefetch.valid()) {
                    start_prefetch(index + 1);
                }
            } else if (streaming && !prefetch_stream && index > 0 && !is_context_cached(index)) {
                stream_bytes += load_stream_context(index);
            }
            if (overlap_htp_preload && index + 1 < (int32_t) contexts.size() &&
                    !htp_resident[index + 1] && !tree_htp_preload.valid()) {
                const bool next_is_last = index + 2 == (int32_t) contexts.size();
                std::vector<int8_t> saved_output(current->logits, current->logits + current->n_tokens);
                for (int32_t i = 0; i < current->n_tokens; ++i) {
                    current->logits[i] = next_is_last ? final_output[i] : true;
                }
                const bool started = start_htp_preload(index + 1, *current, false);
                std::copy(saved_output.begin(), saved_output.end(), current->logits);
                if (!started) {
                    return false;
                }
            }
            const int64_t trace_decode_start = trace_clusters ? ggml_time_us() : 0;
            early_residency_diag(model, il_start, "cluster.after_load", index);
            const int decode_status = llama_decode(ctx, *current);
            trace_cluster(index, "decode_call", trace_decode_start, trace_clusters ? ggml_time_us() : 0);
            if (trace_clusters && decode_status == 0) {
                llama_synchronize(ctx);
                trace_cluster(index, "execute_sync", trace_decode_start, ggml_time_us());
            }
            if (decode_status != 0) {
                return false;
            }
            early_residency_diag(model, il_start, "cluster.after_decode", index);
            if (streaming && prefetch_stream && !direct_mode && index + 1 < (int32_t) contexts.size() &&
                    !htp_resident[index + 1]) {
                finish_prefetch(index + 1);
            }
            if (overlap_htp_preload && index + 1 < (int32_t) contexts.size() && !htp_resident[index + 1]) {
                const bool next_is_last = index + 2 == (int32_t) contexts.size();
                for (int32_t i = 0; i < scratch.n_tokens; ++i) {
                    scratch.logits[i] = next_is_last ? final_output[i] : true;
                }
                if (prefetch.valid() && prefetch_index == index + 1) {
                    finish_prefetch(index + 1);
                }
                if (!finish_tree_htp_preload(index + 1)) {
                    return false;
                }
            }

            if (streaming) {
                const uint64_t n_context_layers = (uint64_t) (il_end - il_begin);
                if (is_context_cached(index) && htp_resident[index]) {
                    cache_hits += n_context_layers;
                } else {
                    layer_loads += n_context_layers;
                }
            }

            if (is_context_cached(index)) {
                htp_resident[index] = 1;
            }

            if (index + 1 == (int32_t) contexts.size()) {
                const int64_t trace_sync_start = trace_clusters ? ggml_time_us() : 0;
                const auto wait_start = std::chrono::steady_clock::now();
                const bool ok = capture_logits(ctx, *current);
                trace_cluster(index, "result_sync", trace_sync_start, trace_clusters ? ggml_time_us() : 0);
                const auto wait_end = std::chrono::steady_clock::now();
                htp_compute_wait_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count();
                if (!is_context_cached(index)) {
                    llama_release_compute(ctx);
                }
                if (streaming && !is_context_cached(index)) {
                    discard_stream_context(index);
                }
                return ok;
            }

            const int64_t trace_sync_start = trace_clusters ? ggml_time_us() : 0;
            const auto wait_start = std::chrono::steady_clock::now();
            float * src = llama_get_embeddings_layer_inp(ctx, il_end);
            trace_cluster(index, "result_sync", trace_sync_start, trace_clusters ? ggml_time_us() : 0);
            const auto wait_end = std::chrono::steady_clock::now();
            htp_compute_wait_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count();
            if (!src) {
                return false;
            }
            hidden.assign(src, src + (size_t) current->n_tokens * n_embd);
            if (std::any_of(hidden.begin(), hidden.end(), [](float value) { return !std::isfinite(value); })) {
                std::fprintf(stderr, "Tail hidden contains non-finite values after layers [%d,%d)\n", il_begin, il_end);
                return false;
            }

            const int32_t n_tokens = current->n_tokens;
            std::vector<llama_pos> positions(current->pos, current->pos + n_tokens);
            std::vector<std::vector<llama_seq_id>> seq_ids(n_tokens);
            for (int32_t i = 0; i < n_tokens; ++i) {
                seq_ids[i].assign(current->seq_id[i], current->seq_id[i] + current->n_seq_id[i]);
            }
            scratch.n_tokens = 0;
            for (int32_t i = 0; i < n_tokens; ++i) {
                const int32_t j = scratch.n_tokens++;
                std::copy(hidden.data() + (size_t) i * n_embd, hidden.data() + (size_t) (i + 1) * n_embd,
                        scratch.embd + (size_t) j * n_embd);
                scratch.pos[j] = positions[i];
                scratch.n_seq_id[j] = (int32_t) seq_ids[i].size();
                for (int32_t s = 0; s < scratch.n_seq_id[j]; ++s) {
                    scratch.seq_id[j][s] = seq_ids[i][s];
                }
                const bool next_is_last = index + 2 == (int32_t) contexts.size();
                scratch.logits[j] = next_is_last ? final_output[i] : true;
            }

            if (!is_context_cached(index)) {
                llama_release_compute(ctx);
            }
            if (streaming && !is_context_cached(index)) {
                discard_stream_context(index);
            }
            current = &scratch;
        }
        return false;
    }

    bool copy_logits(int32_t row, std::vector<float> & dst) const {
        if (row < 0) {
            row += (int32_t) logits.size();
        }
        if (row < 0 || row >= (int32_t) logits.size()) {
            return false;
        }
        dst = logits[row];
        mask_eog_logits(dst);
        return true;
    }

    bool memory_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
        for (auto * ctx : contexts) {
            if (!llama_memory_seq_rm(llama_get_memory(ctx), seq_id, p0, p1)) {
                return false;
            }
        }
        return true;
    }

    void memory_seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) {
        for (auto * ctx : contexts) {
            llama_memory_seq_cp(llama_get_memory(ctx), src, dst, p0, p1);
        }
    }

    void memory_seq_keep(llama_seq_id seq_id) {
        for (auto * ctx : contexts) {
            llama_memory_seq_keep(llama_get_memory(ctx), seq_id);
        }
    }
};

#include "gain-cost-tree.h"

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    options opts;
    try {
        if (!parse_options(argc, argv, opts)) {
            print_usage(argv[0]);
            return 1;
        }
    } catch (...) {
        print_usage(argv[0]);
        return 1;
    }

    if (!opts.prompt_file.empty()) {
        std::ifstream file(opts.prompt_file, std::ios::binary);
        if (!file) {
            std::fprintf(stderr, "failed to open prompt file\n");
            return 1;
        }
        opts.prompt.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }
    if (opts.n_ubatch == 0) {
        opts.n_ubatch = opts.n_batch;
    }

#if !defined(_WIN32)
    if (opts.tail_graph_reuse) {
        setenv("GGML_HEXAGON_GRAPH_PLAN_CACHE", "1", 1);
    }
#endif

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = opts.tail_offload_layers;
    if (opts.tail_ffn_stream || opts.tail_layer_stream) {
        mparams.load_mode = LLAMA_LOAD_MODE_MMAP;
        mparams.tensor_read_lazy = LLAMA_TENSOR_READ_LAZY_ON;
    }

    if (!opts.tail_device.empty()) {
        ggml_backend_load_all();
    }

    ggml_backend_dev_t devices[] = {
        opts.tail_device.empty() ? ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0) :
                ggml_backend_dev_by_name(opts.tail_device.c_str()),
        nullptr,
    };
    if (!devices[0]) {
        std::fprintf(stderr, "requested backend is unavailable: %s\n",
                opts.tail_device.empty() ? "CPU" : opts.tail_device.c_str());
        return 1;
    }
    mparams.devices = devices;

    const llama_model_tensor_buft_override output_overrides[] = {
        { "^(output|output_norm)\\.(weight|bias)$", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    const llama_model_tensor_buft_override stream_overrides[] = {
        { "^(output|output_norm)\\.(weight|bias)$", ggml_backend_cpu_buffer_type() },
        { "^blk\\.[0-9]+\\.ffn_(gate|up|down)\\.weight$", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    std::string htp_resident_pattern;
    std::vector<llama_model_tensor_buft_override> layer_stream_overrides;
    layer_stream_overrides.push_back({ "^(output|output_norm)\\.(weight|bias)$", ggml_backend_cpu_buffer_type() });
    if (opts.tail_stream_cache_layers > 0) {
        htp_resident_pattern = "^blk\\.(";
        for (int32_t il = 0; il < opts.tail_stream_cache_layers; ++il) {
            if (il > 0) {
                htp_resident_pattern += "|";
            }
            htp_resident_pattern += std::to_string(opts.exit_layer + il);
        }
        htp_resident_pattern += ")\\.";
        layer_stream_overrides.push_back({ htp_resident_pattern.c_str(), ggml_backend_dev_buffer_type(devices[0]) });
    }
    layer_stream_overrides.push_back({
            "^blk\\.[0-9]+\\.(attn_norm|attn_(q|k|v|output)|ffn_norm|ffn_(gate|up|down))\\.(weight|bias)$",
            ggml_backend_cpu_buffer_type() });
    layer_stream_overrides.push_back({ nullptr, nullptr });
    if (opts.tail_layer_stream) {
        mparams.tensor_buft_overrides = layer_stream_overrides.data();
        if (opts.tail_stream_cache_layers > 0) {
            std::fprintf(stderr, "layerskip: fixed HTP resident tail=[%d,%d)\n", opts.exit_layer,
                    opts.exit_layer + opts.tail_stream_cache_layers);
        }
    } else if (opts.tail_ffn_stream) {
        mparams.tensor_buft_overrides = stream_overrides;
    } else if (!opts.tail_device.empty() && !opts.tail_head_on_device) {
        mparams.tensor_buft_overrides = output_overrides;
    }

    llama_model * model = llama_model_load_from_file(opts.model.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_embd = llama_model_n_embd(model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    if (opts.ignore_eos) {
        eog_mask.resize(n_vocab);
        int32_t eligible = 0;
        for (int32_t token = 0; token < n_vocab; ++token) {
            eog_mask[token] = llama_vocab_is_eog(vocab, token);
            eligible += !eog_mask[token];
        }
        if (eligible < 2 || (opts.tree_gain_cost && eligible < opts.tree_top_k)) {
            std::fprintf(stderr, "not enough non-EOG tokens for sampling\n");
            llama_model_free(model);
            return 1;
        }
        std::fprintf(stderr, "layerskip: ignore-eos masks all EOG tokens before sampling\n");
    }

    logits_dump dump;
    if (!opts.logits_prefix.empty() && !dump.open(opts.logits_prefix, n_vocab)) {
        std::fprintf(stderr, "failed to open or write logits dump files for prefix: %s\n", opts.logits_prefix.c_str());
        llama_model_free(model);
        return 1;
    }

    if (opts.exit_layer >= n_layer) {
        std::fprintf(stderr, "exit layer must be in [1, %d)\n", n_layer);
        llama_model_free(model);
        return 1;
    }

    if (!opts.tail_device.empty()) {
        const int32_t expected = n_layer - opts.exit_layer + 1;
        if (opts.tail_offload_layers != expected && !opts.tail_head_on_device) {
            std::fprintf(stderr, "tail offload layers must be %d for exit layer %d and %d total layers\n",
                    expected, opts.exit_layer, n_layer);
            llama_model_free(model);
            return 1;
        }
        if (opts.tail_head_on_device) {
            std::fprintf(stderr, "layerskip placement diagnostic: %d output/repeating layers on %s\n",
                    opts.tail_offload_layers, opts.tail_device.c_str());
        } else {
            std::fprintf(stderr, "layerskip placement: early=[0,%d) CPU, tail=[%d,%d) %s, head weights CPU\n",
                    opts.exit_layer, opts.exit_layer, n_layer, opts.tail_device.c_str());
        }
    }

    const int32_t n_prompt = -llama_tokenize(vocab, opts.prompt.c_str(), opts.prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> prompt(n_prompt);
    if (n_prompt <= 0 || llama_tokenize(vocab, opts.prompt.c_str(), opts.prompt.size(), prompt.data(), n_prompt, true, true) < 0) {
        std::fprintf(stderr, "failed to tokenize prompt\n");
        llama_model_free(model);
        return 1;
    }

    if (n_prompt + opts.n_predict > opts.n_ctx || n_prompt > opts.n_batch) {
        std::fprintf(stderr, "prompt or generation does not fit context/batch\n");
        llama_model_free(model);
        return 1;
    }

    using clock = std::chrono::steady_clock;

    if (opts.target_only) {
        llama_context_params params = llama_context_default_params();
        params.n_ctx = opts.n_ctx;
        params.n_batch = opts.n_batch;
        params.n_ubatch = opts.n_ubatch;
        params.n_threads = opts.n_threads;
        params.n_threads_batch = opts.n_threads;
        params.no_perf = false;
        params.offload_kqv = false;
        if (opts.flash_attn_off) {
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }

        llama_context * ctx = llama_init_from_model(model, params);
        if (!ctx) {
            std::fprintf(stderr, "failed to create target context\n");
            llama_model_free(model);
            return 1;
        }

        std::fprintf(stderr, "layerskip: target-only, full=[0,%d)\n", n_layer);
        for (llama_token token : prompt) {
            print_token(vocab, token);
        }

        llama_batch batch = llama_batch_init(opts.n_batch, 0, 1);
        for (int32_t i = 0; i < n_prompt; ++i) {
            batch_add_token(batch, prompt[i], i, i == n_prompt - 1);
        }

        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "target prompt decode failed\n");
            llama_batch_free(batch);
            llama_free(ctx);
            llama_model_free(model);
            return 2;
        }
        const auto generation_start = clock::now();

        std::vector<llama_token> generated;
        generated.reserve(opts.n_predict);
        for (int32_t i = 0; i < opts.n_predict; ++i) {
            if (!opts.logits_prefix.empty() && !dump.write(vocab, llama_get_logits_ith(ctx, -1), n_vocab, i, n_prompt + i)) {
                std::fprintf(stderr, "failed to write logits dump\n");
                llama_batch_free(batch);
                llama_free(ctx);
                llama_model_free(model);
                return 2;
            }
            const llama_token token = argmax_token(llama_get_logits_ith(ctx, -1), n_vocab);
            print_token(vocab, token);
            generated.push_back(token);
            if (!opts.ignore_eos && llama_vocab_is_eog(vocab, token)) {
                break;
            }
            if (i + 1 == opts.n_predict) {
                break;
            }

            batch.n_tokens = 0;
            batch_add_token(batch, token, n_prompt + i, true);
            if (llama_decode(ctx, batch) != 0) {
                std::fprintf(stderr, "target token decode failed\n");
                llama_batch_free(batch);
                llama_free(ctx);
                llama_model_free(model);
                return 2;
            }
        }

        const auto time_end = clock::now();
        const double generation_seconds = std::chrono::duration<double>(time_end - generation_start).count();
        std::fprintf(stderr, "\nlayerskip target-only generation %zu tokens in %.3f seconds, speed: %.3f t/s\n",
                generated.size(), generation_seconds, generated.size() / generation_seconds);
        print_token_trace(generated);

        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    }

    llama_context_params early_params = llama_context_default_params();
    early_params.n_ctx = opts.n_ctx;
    early_params.n_batch = opts.n_batch;
    early_params.n_ubatch = opts.n_ubatch;
    early_params.n_seq_max = opts.tree_gain_cost ? opts.tree_max_nodes + 1 : opts.draft_tree ? opts.draft_max + 1 : 1;
    early_params.kv_unified = opts.draft_tree;
    early_params.n_threads = opts.n_threads;
    early_params.n_threads_batch = opts.n_threads;
    early_params.n_layer_start = 0;
    early_params.n_layer_end = opts.exit_layer;
    early_params.no_perf = false;
    early_params.offload_kqv = false;
    if (opts.flash_attn_off) {
        early_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    llama_context_params tail_params = early_params;
    tail_params.n_layer_start = opts.exit_layer;
    tail_params.n_layer_end = n_layer;

    llama_context * ctx_early = llama_init_from_model(model, early_params);
    if (opts.tail_stream_cache_layers > n_layer - opts.exit_layer) {
        std::fprintf(stderr, "tail stream cache layers must be in [0, %d]\n", n_layer - opts.exit_layer);
        llama_free(ctx_early);
        llama_model_free(model);
        return 1;
    }
    if (!opts.tail_stream_cluster_sizes.empty()) {
        int64_t specified_layers = 0;
        for (int32_t size : opts.tail_stream_cluster_sizes) {
            specified_layers += size;
        }
        const int32_t remaining_layers = n_layer - opts.exit_layer - opts.tail_stream_cache_layers;
        if (specified_layers != remaining_layers) {
            std::fprintf(stderr, "tail stream cluster sizes cover %lld layers, expected %d\n",
                    (long long) specified_layers, remaining_layers);
            llama_free(ctx_early);
            llama_model_free(model);
            return 1;
        }
    }

    const int32_t tail_stream_slots = opts.tail_stream_slots > 0 ? opts.tail_stream_slots :
            (!opts.tail_stream_cluster_sizes.empty() || opts.tail_stream_cluster_layers > 1 ? 1 :
                    (opts.tail_stream_prefetch ? 2 : 1));
    const bool fixed_htp_cache = opts.tail_layer_stream && opts.tail_stream_cache_layers > 0;
    tail_decoder tail(model, opts.exit_layer, n_layer, n_embd, n_vocab, tail_params,
            opts.tail_layerwise || opts.tail_ffn_stream || opts.tail_layer_stream,
            opts.tail_ffn_stream, opts.tail_layer_stream, opts.tail_stream_prefetch,
            opts.tail_graph_reuse, opts.tail_stream_cache_layers, fixed_htp_cache,
            opts.tail_stream_cluster_layers, opts.tail_stream_cluster_sizes, tail_stream_slots);
    if (opts.memory_budget_mib > 0) {
        const uint64_t mib = 1024ULL * 1024ULL;
        const uint64_t budget_bytes = (uint64_t) opts.memory_budget_mib * mib;
        const uint64_t runtime_bytes = (uint64_t) opts.runtime_reserve_mib * mib;
        uint64_t all_layer_bytes = 0;
        uint64_t early_bytes = 0;
        std::vector<uint64_t> tail_bytes;
        for (int32_t il = 0; il < n_layer; ++il) {
            const uint64_t bytes = tail.stream_layer_bytes(il);
            all_layer_bytes += bytes;
            if (il < opts.exit_layer) {
                early_bytes += bytes;
            } else {
                tail_bytes.push_back(bytes);
            }
        }

        const uint64_t model_bytes = llama_model_size(model);
        if (model_bytes < all_layer_bytes) {
            std::fprintf(stderr, "memory plan failed: layer bytes exceed model bytes\n");
            llama_free(ctx_early);
            llama_model_free(model);
            return 1;
        }

        const uint64_t fixed_bytes = model_bytes - all_layer_bytes;
        uint64_t planned_bytes = 0;
        uint64_t stage_bytes = 0;
        int32_t cache_layers = -1;
        const bool fixed_cache_count = fixed_htp_cache || !opts.tail_stream_cluster_sizes.empty();
        const int32_t first_count = fixed_cache_count ? opts.tail_stream_cache_layers : 0;
        const int32_t last_count = fixed_cache_count ? opts.tail_stream_cache_layers : (int32_t) tail_bytes.size();
        for (int32_t count = first_count; count <= last_count; ++count) {
            uint64_t cached_bytes = 0;
            for (int32_t i = 0; i < count; ++i) {
                cached_bytes += tail_bytes[i];
            }
            uint64_t current_stage_bytes = 0;
            if (!opts.tail_stream_cluster_sizes.empty()) {
                int32_t il = count;
                for (int32_t size : opts.tail_stream_cluster_sizes) {
                    uint64_t cluster_bytes = 0;
                    for (int32_t j = il; j < il + size; ++j) {
                        cluster_bytes += tail_bytes[j];
                    }
                    current_stage_bytes = std::max(current_stage_bytes, cluster_bytes);
                    il += size;
                }
            } else {
                for (int32_t i = count; i < (int32_t) tail_bytes.size(); i += opts.tail_stream_cluster_layers) {
                    uint64_t cluster_bytes = 0;
                    const int32_t cluster_end = std::min((int32_t) tail_bytes.size(), i + opts.tail_stream_cluster_layers);
                    for (int32_t j = i; j < cluster_end; ++j) {
                        cluster_bytes += tail_bytes[j];
                    }
                    current_stage_bytes = std::max(current_stage_bytes, cluster_bytes);
                }
            }
            const uint64_t stream_buffer_bytes = current_stage_bytes * tail_stream_slots;
            const uint64_t current_plan_bytes = fixed_bytes + early_bytes + cached_bytes + stream_buffer_bytes + runtime_bytes;
            if (current_plan_bytes <= budget_bytes) {
                cache_layers = count;
                planned_bytes = current_plan_bytes;
                stage_bytes = current_stage_bytes;
            }
        }

        if (cache_layers < 0) {
            std::fprintf(stderr, "memory plan failed: budget %d MiB cannot hold fixed weights, early layers, stream buffers and runtime reserve\n",
                    opts.memory_budget_mib);
            llama_free(ctx_early);
            llama_model_free(model);
            return 1;
        }

        opts.tail_stream_cache_layers = cache_layers;
        tail.cache_layers = cache_layers;
        const uint64_t stream_buffer_bytes = stage_bytes * tail_stream_slots;
        const uint64_t selected_cache_bytes = planned_bytes - fixed_bytes - early_bytes - stream_buffer_bytes - runtime_bytes;
        std::fprintf(stderr, "memory plan budget = %llu bytes\n", (unsigned long long) budget_bytes);
        std::fprintf(stderr, "memory plan fixed weights = %llu bytes\n", (unsigned long long) fixed_bytes);
        std::fprintf(stderr, "memory plan early weights = %llu bytes\n", (unsigned long long) early_bytes);
        std::fprintf(stderr, "memory plan cached tail = %llu bytes / %d layers\n",
                (unsigned long long) selected_cache_bytes, cache_layers);
        std::fprintf(stderr, "memory plan cached tail placement = %s\n", fixed_htp_cache ? "HTP fixed buffer" : "CPU mmap");
        if (opts.tail_stream_cluster_sizes.empty()) {
            std::fprintf(stderr, "memory plan stream buffers = %llu bytes / %d slots x %d layers\n",
                    (unsigned long long) stream_buffer_bytes, tail_stream_slots, opts.tail_stream_cluster_layers);
        } else {
            std::fprintf(stderr, "memory plan stream buffers = %llu bytes / %d slots x mixed groups (max %llu bytes)\n",
                    (unsigned long long) stream_buffer_bytes, tail_stream_slots, (unsigned long long) stage_bytes);
        }
        std::fprintf(stderr, "memory plan runtime reserve = %llu bytes\n", (unsigned long long) runtime_bytes);
        std::fprintf(stderr, "memory plan total = %llu bytes\n", (unsigned long long) planned_bytes);
    }

    if (!ctx_early || !tail.init()) {
        std::fprintf(stderr, "failed to create split contexts\n");
        tail.clear();
        llama_free(ctx_early);
        llama_model_free(model);
        return 1;
    }

    llama_set_embeddings_layer_inp(ctx_early, opts.exit_layer, true);

    std::fprintf(stderr, "layerskip: one model=%p, early=[0,%d), tail=[%d,%d), draft=%d\n",
            (void *) model, opts.exit_layer, opts.exit_layer, n_layer, opts.draft_max);
    if (opts.tail_layer_stream) {
        if (opts.tail_stream_cluster_sizes.empty()) {
            std::fprintf(stderr, "layerskip: whole tail layer streaming, cluster=%d, contexts=%zu\n",
                    opts.tail_stream_cluster_layers, tail.contexts.size());
        } else {
            std::fprintf(stderr, "layerskip: whole tail layer streaming, mixed clusters, contexts=%zu\n",
                    tail.contexts.size());
            for (size_t i = 0; i < tail.contexts.size(); ++i) {
                std::fprintf(stderr, "layerskip: tail context [%d,%d) %s\n", tail.context_starts[i],
                        tail.context_ends[i], tail.is_context_cached((int32_t) i) ? "cached" : "streamed");
            }
        }
    } else if (opts.tail_ffn_stream) {
        std::fprintf(stderr, "layerskip: tail FFN naive streaming, %d one-layer contexts\n", n_layer - opts.exit_layer);
    } else if (opts.tail_layerwise) {
        std::fprintf(stderr, "layerskip: resident tail layerwise diagnostic, %d one-layer contexts\n", n_layer - opts.exit_layer);
    }

    for (llama_token token : prompt) {
        print_token(vocab, token);
    }

    const int32_t batch_seq_max = opts.tree_gain_cost ? opts.tree_max_nodes + 1 : opts.draft_tree ? opts.draft_max + 1 : 1;
    llama_batch batch_early = llama_batch_init(std::max(opts.n_batch, opts.tree_max_nodes + 1), 0, batch_seq_max);
    for (int32_t i = 0; i < n_prompt; ++i) {
        batch_add_token(batch_early, prompt[i], i, i == n_prompt - 1);
    }

    auto elapsed_ms = [](clock::time_point start) {
        return std::chrono::duration<double, std::milli>(clock::now() - start).count();
    };

    double t_prompt_early_ms = 0.0;
    double t_prompt_tail_ms = 0.0;
    double t_draft_build_ms = 0.0;
    double t_tail_verify_ms = 0.0;
    double t_match_ms = 0.0;
    double t_last_early_ms = 0.0;
    double t_last_tail_ms = 0.0;
    double t_bonus_early_ms = 0.0;
    double t_rollback_ms = 0.0;
    double t_correction_early_ms = 0.0;

    const auto time_start = clock::now();
    const auto prompt_early_start = clock::now();
    if (llama_decode(ctx_early, batch_early) != 0) {
        std::fprintf(stderr, "early prompt decode failed\n");
        return 2;
    }

    float * prompt_hidden_ptr = llama_get_embeddings_layer_inp(ctx_early, opts.exit_layer);
    std::vector<float> prompt_hidden(prompt_hidden_ptr, prompt_hidden_ptr + (size_t) n_prompt * n_embd);
    std::vector<float> early_logits;
    if (!copy_logits(ctx_early, -1, n_vocab, early_logits)) {
        std::fprintf(stderr, "missing early logits\n");
        return 2;
    }
    t_prompt_early_ms += elapsed_ms(prompt_early_start);

    const auto prompt_tail_start = clock::now();
    llama_batch batch_tail = llama_batch_init(std::max(opts.n_batch, opts.tree_max_nodes + 1), n_embd, batch_seq_max);
    const bool prompt_tail_direct = opts.tail_layer_stream && opts.tail_stream_prefetch;
    const int32_t prompt_tail_batch = prompt_tail_direct ? std::min(opts.n_batch, opts.n_ubatch) : opts.n_batch;
    for (int32_t begin = 0; begin < n_prompt; begin += prompt_tail_batch) {
        batch_tail.n_tokens = 0;
        const int32_t end = std::min(begin + prompt_tail_batch, n_prompt);
        for (int32_t i = begin; i < end; ++i) {
            batch_add_embd(batch_tail, prompt_hidden.data() + (size_t) i * n_embd, n_embd, i, i == end - 1);
        }
        if (!tail.decode(batch_tail)) {
            std::fprintf(stderr, "tail prompt decode failed\n");
            return 2;
        }
    }
    if (prompt_tail_direct) {
        std::fprintf(stderr, "layerskip: Tail prefill direct batches=%d, max-tokens=%d\n",
                (n_prompt + prompt_tail_batch - 1) / prompt_tail_batch, prompt_tail_batch);
    }

    std::vector<float> target_logits;
    if (!tail.copy_logits(-1, target_logits)) {
        std::fprintf(stderr, "missing target logits\n");
        return 2;
    }
    t_prompt_tail_ms += elapsed_ms(prompt_tail_start);
    tail.trace_generation = true;
    if (opts.tree_gain_cost) {
        if (opts.tree_top_k > n_vocab) {
            std::fprintf(stderr, "tree top-k exceeds vocabulary size\n");
            return 1;
        }
        const int result = run_gain_tree(opts, ctx_early, tail, vocab, n_vocab, n_embd, n_prompt,
                batch_early, batch_tail, early_logits, target_logits, t_prompt_early_ms, t_prompt_tail_ms);
        llama_batch_free(batch_tail);
        llama_batch_free(batch_early);
        tail.clear();
        llama_free(ctx_early);
        llama_model_free(model);
        return result;
    }
    const auto generation_start = clock::now();

    if (opts.split_target_only) {
        std::fprintf(stderr, "layerskip: split-target-only, early=[0,%d), tail=[%d,%d)\n",
                opts.exit_layer, opts.exit_layer, n_layer);
        std::vector<llama_token> generated;
        generated.reserve(opts.n_predict);

        for (int32_t i = 0; i < opts.n_predict; ++i) {
            if (!opts.logits_prefix.empty() && !dump.write(vocab, target_logits.data(), n_vocab, i, n_prompt + i)) {
                std::fprintf(stderr, "failed to write logits dump\n");
                llama_batch_free(batch_tail);
                llama_batch_free(batch_early);
                tail.clear();
                llama_free(ctx_early);
                llama_model_free(model);
                return 2;
            }
            const llama_token token = argmax_token(target_logits.data(), n_vocab);
            print_token(vocab, token);
            generated.push_back(token);
            if (!opts.ignore_eos && llama_vocab_is_eog(vocab, token)) {
                break;
            }
            if (i + 1 == opts.n_predict) {
                break;
            }

            const llama_pos pos = n_prompt + i;
            batch_early.n_tokens = 0;
            batch_add_token(batch_early, token, pos, true);
            if (llama_decode(ctx_early, batch_early) != 0) {
                std::fprintf(stderr, "split target early decode failed\n");
                return 2;
            }

            float * hidden = llama_get_embeddings_layer_inp(ctx_early, opts.exit_layer);
            batch_tail.n_tokens = 0;
            batch_add_embd(batch_tail, hidden, n_embd, pos, true);
            if (!tail.decode(batch_tail) || !tail.copy_logits(-1, target_logits)) {
                std::fprintf(stderr, "split target tail decode failed\n");
                return 2;
            }
        }

        const auto time_end = clock::now();
        const double seconds = std::chrono::duration<double>(time_end - time_start).count();
        const double generation_seconds = std::chrono::duration<double>(time_end - generation_start).count();
        std::fprintf(stderr, "\nlayerskip split-target-only decoded %zu tokens in %.3f seconds, speed: %.3f t/s\n",
                generated.size(), seconds, generated.size() / seconds);
        std::fprintf(stderr, "layerskip split-target-only generation %zu tokens in %.3f seconds, speed: %.3f t/s\n",
                generated.size(), generation_seconds, generated.size() / generation_seconds);
        if (opts.tail_ffn_stream || opts.tail_layer_stream) {
            std::fprintf(stderr, "tail stream layer loads = %llu\n", (unsigned long long) tail.layer_loads);
            std::fprintf(stderr, opts.tail_layer_stream ? "tail stream weight bytes = %llu\n" : "tail stream FFN bytes   = %llu\n",
                    (unsigned long long) tail.stream_bytes);
            std::fprintf(stderr, "tail stream prefetch calls = %llu\n", (unsigned long long) tail.prefetch_calls);
            std::fprintf(stderr, "tail stream prefetch failures = %llu\n", (unsigned long long) tail.prefetch_failures);
            std::fprintf(stderr, "tail stream prefetch time = %.3f ms\n", tail.prefetch_time_us / 1000.0);
            std::fprintf(stderr, "tail stream prefetch wait = %.3f ms\n", tail.prefetch_wait_us / 1000.0);
            std::fprintf(stderr, "tail HTP preload calls = %llu\n", (unsigned long long) tail.htp_preload_calls);
            std::fprintf(stderr, "tail HTP preload failures = %llu\n", (unsigned long long) tail.htp_preload_failures);
            std::fprintf(stderr, "tail HTP preload bytes = %llu\n", (unsigned long long) tail.htp_preload_bytes);
            std::fprintf(stderr, "tail HTP preload time = %.3f ms\n", tail.htp_preload_time_us / 1000.0);
            std::fprintf(stderr, "tail HTP direct loads = %llu\n", (unsigned long long) tail.htp_direct_loads);
            std::fprintf(stderr, "tail HTP direct bytes = %llu\n", (unsigned long long) tail.htp_direct_bytes);
            std::fprintf(stderr, "tail HTP compute wait = %.3f ms\n", tail.htp_compute_wait_us / 1000.0);
            std::fprintf(stderr, "tail stream discard failures = %llu\n", (unsigned long long) tail.discard_failures);
            std::fprintf(stderr, "tail stream cache layers = %d\n", opts.tail_stream_cache_layers);
            std::fprintf(stderr, "tail stream cache hits = %llu\n", (unsigned long long) tail.cache_hits);
            std::fprintf(stderr, "tail stream cache bytes = %llu\n", (unsigned long long) tail.cache_bytes);
        }
        print_token_trace(generated);

        llama_batch_free(batch_tail);
        llama_batch_free(batch_early);
        tail.clear();
        llama_free(ctx_early);
        llama_model_free(model);
        return 0;
    }

    int32_t n_generated = 0;
    int32_t n_drafted = 0;
    int32_t n_accept = 0;
    int32_t n_verify = 0;
    int32_t n_bonus = 0;
    int32_t n_last_completed = 0;
    int32_t n_last_skipped = 0;
    std::vector<llama_token> generated;
    generated.reserve(opts.n_predict);
    bool has_pending_tail = false;
    llama_pos pending_tail_pos = -1;
    std::vector<float> pending_tail_hidden;
    bool stop = false;

    int32_t n_tree_branches = 0;
    int32_t n_tree_nodes = 0;
    int32_t n_side_entries = 0;
    int32_t n_side_accept = 0;
    int32_t n_tree_capacity_skipped = 0;

    if (opts.draft_tree) {
        std::fprintf(stderr, "layerskip: draft tree, max-nodes=%d, side=full, single=%d, p-ratio=%.3f\n",
                opts.tree_max_nodes, opts.tree_single_branch, opts.tree_p_ratio);

        while (n_generated < opts.n_predict && !stop) {
            if (opts.tree_tail_prefetch_overlap) {
                tail.start_tree_prefetch();
            }
            const int32_t n_round = std::min(opts.draft_max, opts.n_predict - n_generated);
            const llama_pos pos_base = n_prompt + n_generated;

            struct side_branch {
                llama_seq_id seq_id = -1;
                int32_t split_depth = -1;
                std::vector<llama_token> tokens;
                std::vector<float> hidden;
                std::vector<float> logits;
                std::vector<int32_t> rows;
            };

            std::vector<llama_token> main_tokens;
            std::vector<float> main_hidden;
            std::vector<side_branch> branches;
            main_tokens.reserve(n_round);
            main_hidden.reserve((size_t) n_round * n_embd);
            branches.reserve(n_round);

            int32_t planned_nodes = n_round;
            bool tail_tree_prepared = false;
            auto phase_start = clock::now();
            for (int32_t depth = 0; depth < n_round; ++depth) {
                const greedy_pair candidates = top2_tokens(early_logits.data(), n_vocab);
                if ((!opts.tree_single_branch || branches.empty()) &&
                        candidates.second != LLAMA_TOKEN_NULL && candidates.ratio >= opts.tree_p_ratio) {
                    const int32_t branch_nodes = n_round - depth;
                    if (planned_nodes + branch_nodes <= opts.tree_max_nodes) {
                        side_branch branch;
                        branch.seq_id = (llama_seq_id) branches.size() + 1;
                        branch.split_depth = depth;
                        branch.tokens.push_back(candidates.second);
                        branch.hidden.reserve((size_t) branch_nodes * n_embd);
                        branch.rows.assign(branch_nodes, -1);
                        branches.push_back(std::move(branch));
                        planned_nodes += branch_nodes;
                    } else {
                        ++n_tree_capacity_skipped;
                    }
                }

                main_tokens.push_back(candidates.first);
                batch_early.n_tokens = 0;
                batch_add_token_seqs(batch_early, candidates.first, pos_base + depth, 0, -1, true);
                if (llama_decode(ctx_early, batch_early) != 0) {
                    std::fprintf(stderr, "early main branch decode failed\n");
                    return 2;
                }
                float * hidden = llama_get_embeddings_layer_inp(ctx_early, opts.exit_layer);
                main_hidden.insert(main_hidden.end(), hidden, hidden + n_embd);
                if (!copy_logits(ctx_early, 0, n_vocab, early_logits)) {
                    return 2;
                }
            }

            auto prepare_tail_sequences = [&]() -> bool {
                for (const side_branch & branch : branches) {
                    if (!tail.memory_seq_rm(branch.seq_id, -1, -1)) {
                        return false;
                    }
                    tail.memory_seq_cp(0, branch.seq_id, -1, -1);
                }
                return true;
            };

            auto add_tail_layout = [&](bool zero_sides, std::vector<int32_t> * main_rows) {
                std::vector<llama_seq_id> seq_ids;
                seq_ids.reserve(branches.size() + 1);
                std::vector<float> zero_hidden(n_embd, 0.0f);
                if (has_pending_tail) {
                    seq_ids.push_back(0);
                    for (const side_branch & branch : branches) {
                        seq_ids.push_back(branch.seq_id);
                    }
                    batch_add_embd_seq_list(batch_tail, pending_tail_hidden.data(), n_embd,
                            pending_tail_pos, seq_ids, true);
                }
                for (int32_t depth = 0; depth < n_round; ++depth) {
                    seq_ids.clear();
                    seq_ids.push_back(0);
                    for (const side_branch & branch : branches) {
                        if (depth < branch.split_depth) {
                            seq_ids.push_back(branch.seq_id);
                        }
                    }
                    if (main_rows) {
                        (*main_rows)[depth] = batch_tail.n_tokens;
                    }
                    batch_add_embd_seq_list(batch_tail, main_hidden.data() + (size_t) depth * n_embd,
                            n_embd, pos_base + depth, seq_ids, true);
                    for (side_branch & branch : branches) {
                        if (depth >= branch.split_depth) {
                            const int32_t side_index = depth - branch.split_depth;
                            branch.rows[side_index] = batch_tail.n_tokens;
                            const float * side_hidden = zero_sides ? zero_hidden.data() :
                                    branch.hidden.data() + (size_t) side_index * n_embd;
                            batch_add_embd_seqs(batch_tail, side_hidden, n_embd, pos_base + depth,
                                    branch.seq_id, -1, true);
                        }
                    }
                }
            };

            auto try_start_tree_htp_preload = [&](bool wait_for_prefetch) -> bool {
                if (!opts.tree_tail_htp_overlap || branches.empty() || tail_tree_prepared) {
                    return true;
                }
                if (opts.tree_tail_prefetch_overlap) {
                    if (!wait_for_prefetch && !tail.tree_prefetch_ready()) {
                        return true;
                    }
                    tail.finish_tree_prefetch();
                }
                if (!prepare_tail_sequences()) {
                    return false;
                }
                batch_tail.n_tokens = 0;
                add_tail_layout(true, nullptr);
                const bool started = tail.start_tree_htp_preload(batch_tail);
                batch_tail.n_tokens = 0;
                tail_tree_prepared = started;
                return started;
            };

            if (!try_start_tree_htp_preload(false)) {
                std::fprintf(stderr, "tail tree sequence reset failed\n");
                return 2;
            }

            for (side_branch & branch : branches) {
                if (!llama_memory_seq_rm(llama_get_memory(ctx_early), branch.seq_id, -1, -1)) {
                    std::fprintf(stderr, "early tree sequence reset failed\n");
                    return 2;
                }
                llama_memory_seq_cp(llama_get_memory(ctx_early), 0, branch.seq_id, -1, -1);
                if (!llama_memory_seq_rm(llama_get_memory(ctx_early), branch.seq_id,
                        pos_base + branch.split_depth, -1)) {
                    std::fprintf(stderr, "early tree prefix trim failed\n");
                    return 2;
                }
                const int32_t branch_nodes = n_round - branch.split_depth;
                for (int32_t side_index = 0; side_index < branch_nodes; ++side_index) {
                    const llama_token side_token = side_index == 0 ? branch.tokens[0] :
                            argmax_token(branch.logits.data(), n_vocab);
                    if (side_index > 0) {
                        branch.tokens.push_back(side_token);
                    }
                    batch_early.n_tokens = 0;
                    batch_add_token_seqs(batch_early, side_token,
                            pos_base + branch.split_depth + side_index, branch.seq_id, -1, true);
                    if (llama_decode(ctx_early, batch_early) != 0) {
                        std::fprintf(stderr, "early side branch decode failed\n");
                        return 2;
                    }
                    float * hidden = llama_get_embeddings_layer_inp(ctx_early, opts.exit_layer);
                    branch.hidden.insert(branch.hidden.end(), hidden, hidden + n_embd);
                    if (!copy_logits(ctx_early, 0, n_vocab, branch.logits)) {
                        return 2;
                    }
                    if (!try_start_tree_htp_preload(false)) {
                        std::fprintf(stderr, "tail tree sequence reset failed\n");
                        return 2;
                    }
                }
            }
            if (!try_start_tree_htp_preload(true)) {
                std::fprintf(stderr, "tail tree sequence reset failed\n");
                return 2;
            }
            t_draft_build_ms += elapsed_ms(phase_start);

            phase_start = clock::now();
            if (!branches.empty() && !tail_tree_prepared) {
                if (!prepare_tail_sequences()) {
                    std::fprintf(stderr, "tail tree sequence reset failed\n");
                    return 2;
                }
            }

            batch_tail.n_tokens = 0;
            const bool had_pending_tail = has_pending_tail;
            int32_t pending_row = -1;
            if (had_pending_tail) {
                if (pending_tail_pos != pos_base - 1) {
                    std::fprintf(stderr, "pending tail position mismatch\n");
                    return 2;
                }
                pending_row = batch_tail.n_tokens;
            }

            std::vector<int32_t> main_rows(n_round, -1);
            add_tail_layout(false, &main_rows);
            has_pending_tail = false;
            if (!tail.decode(batch_tail)) {
                std::fprintf(stderr, "tail tree verify decode failed\n");
                return 2;
            }

            std::vector<std::vector<float>> verify_logits(batch_tail.n_tokens);
            for (int32_t row = 0; row < batch_tail.n_tokens; ++row) {
                if (!tail.copy_logits(row, verify_logits[row])) {
                    return 2;
                }
            }
            t_tail_verify_ms += elapsed_ms(phase_start);

            ++n_verify;
            const int32_t tree_nodes = planned_nodes;
            n_drafted += tree_nodes;
            n_tree_nodes += tree_nodes;
            n_tree_branches += (int32_t) branches.size();
            int32_t accepted = 0;
            int32_t active_seq = 0;
            int32_t active_branch = -1;
            llama_token correction = LLAMA_TOKEN_NULL;
            bool path_complete = false;
            phase_start = clock::now();
            std::vector<float> current_logits = had_pending_tail ? verify_logits[pending_row] : target_logits;
            for (int32_t depth = 0;; ++depth) {
                if (depth >= n_round) {
                    path_complete = true;
                    break;
                }

                const llama_token expected = argmax_token(current_logits.data(), n_vocab);
                llama_token token = main_tokens[depth];
                int32_t row = main_rows[depth];
                if (active_branch < 0) {
                    for (int32_t i = 0; i < (int32_t) branches.size(); ++i) {
                        if (branches[i].split_depth == depth && branches[i].tokens[0] == expected) {
                            active_branch = i;
                            active_seq = branches[i].seq_id;
                            ++n_side_entries;
                            token = branches[i].tokens[0];
                            row = branches[i].rows[0];
                            break;
                        }
                    }
                } else {
                    side_branch & branch = branches[active_branch];
                    const int32_t side_index = depth - branch.split_depth;
                    token = branch.tokens[side_index];
                    row = branch.rows[side_index];
                }

                if (token != expected) {
                    correction = expected;
                    break;
                }

                print_token(vocab, token);
                generated.push_back(token);
                ++accepted;
                ++n_generated;
                if (active_branch >= 0) {
                    ++n_side_accept;
                }
                target_logits = verify_logits[row];
                current_logits = target_logits;
                if (!opts.ignore_eos && llama_vocab_is_eog(vocab, token)) {
                    stop = true;
                    break;
                }
                if (n_generated >= opts.n_predict) {
                    break;
                }
            }
            n_accept += accepted;
            t_match_ms += elapsed_ms(phase_start);

            if (stop || n_generated >= opts.n_predict) {
                continue;
            }

            const llama_pos reject_pos = pos_base + accepted;
            phase_start = clock::now();
            if (!path_complete) {
                if (!llama_memory_seq_rm(llama_get_memory(ctx_early), active_seq, reject_pos, -1) ||
                    !tail.memory_seq_rm(active_seq, reject_pos, -1)) {
                    std::fprintf(stderr, "tree KV rollback failed\n");
                    return 2;
                }
            }
            if (active_seq != 0) {
                llama_memory_seq_keep(llama_get_memory(ctx_early), active_seq);
                llama_memory_seq_cp(llama_get_memory(ctx_early), active_seq, 0, -1, -1);
                tail.memory_seq_keep(active_seq);
                tail.memory_seq_cp(active_seq, 0, -1, -1);
            }
            llama_memory_seq_keep(llama_get_memory(ctx_early), 0);
            tail.memory_seq_keep(0);
            t_rollback_ms += elapsed_ms(phase_start);

            if (path_complete) {
                ++n_last_completed;
                phase_start = clock::now();
                const llama_token bonus = argmax_token(target_logits.data(), n_vocab);
                const llama_pos bonus_pos = n_prompt + n_generated;
                print_token(vocab, bonus);
                generated.push_back(bonus);
                ++n_generated;
                ++n_bonus;
                if (!opts.ignore_eos && llama_vocab_is_eog(vocab, bonus)) {
                    stop = true;
                    t_bonus_early_ms += elapsed_ms(phase_start);
                    continue;
                }
                if (n_generated >= opts.n_predict) {
                    t_bonus_early_ms += elapsed_ms(phase_start);
                    continue;
                }

                batch_early.n_tokens = 0;
                batch_add_token(batch_early, bonus, bonus_pos, true);
                if (llama_decode(ctx_early, batch_early) != 0) {
                    return 2;
                }
                float * bonus_hidden = llama_get_embeddings_layer_inp(ctx_early, opts.exit_layer);
                pending_tail_hidden.assign(bonus_hidden, bonus_hidden + n_embd);
                pending_tail_pos = bonus_pos;
                has_pending_tail = true;
                if (!copy_logits(ctx_early, -1, n_vocab, early_logits)) {
                    return 2;
                }
                t_bonus_early_ms += elapsed_ms(phase_start);
                continue;
            }

            ++n_last_skipped;
            phase_start = clock::now();
            print_token(vocab, correction);
            generated.push_back(correction);
            ++n_generated;
            if (!opts.ignore_eos && llama_vocab_is_eog(vocab, correction)) {
                stop = true;
                t_correction_early_ms += elapsed_ms(phase_start);
                continue;
            }
            if (n_generated >= opts.n_predict) {
                t_correction_early_ms += elapsed_ms(phase_start);
                continue;
            }

            batch_early.n_tokens = 0;
            batch_add_token(batch_early, correction, reject_pos, true);
            if (llama_decode(ctx_early, batch_early) != 0) {
                return 2;
            }
            float * correction_hidden = llama_get_embeddings_layer_inp(ctx_early, opts.exit_layer);
            if (!copy_logits(ctx_early, -1, n_vocab, early_logits)) {
                return 2;
            }
            pending_tail_hidden.assign(correction_hidden, correction_hidden + n_embd);
            pending_tail_pos = reject_pos;
            has_pending_tail = true;
            t_correction_early_ms += elapsed_ms(phase_start);
        }
    } else {

    while (n_generated < opts.n_predict && !stop) {
        const int32_t n_round = std::min(opts.draft_max, opts.n_predict - n_generated);
        const llama_pos pos_base = n_prompt + n_generated;
        std::vector<llama_token> draft;
        std::vector<float> draft_hidden;
        draft.reserve(n_round);
        draft_hidden.reserve((size_t) n_round * n_embd);

        auto phase_start = clock::now();
        for (int32_t i = 0; i < n_round; ++i) {
            const llama_token token = argmax_token(early_logits.data(), n_vocab);
            draft.push_back(token);

            batch_early.n_tokens = 0;
            batch_add_token(batch_early, token, pos_base + i, true);
            if (early_diag_call("draft.decode", [&]() { return llama_decode(ctx_early, batch_early); }) != 0) {
                std::fprintf(stderr, "early token decode failed\n");
                return 2;
            }

            float * hidden = early_diag_call("draft.hidden_sync", [&]() { return llama_get_embeddings_layer_inp(ctx_early, opts.exit_layer); });
            draft_hidden.insert(draft_hidden.end(), hidden, hidden + n_embd);
            if (!early_diag_call("draft.logits", [&]() { return copy_logits(ctx_early, -1, n_vocab, early_logits); })) {
                return 2;
            }
        }
        t_draft_build_ms += elapsed_ms(phase_start);

        phase_start = clock::now();
        batch_tail.n_tokens = 0;
        const bool had_pending_tail = has_pending_tail;
        if (had_pending_tail) {
            if (pending_tail_pos != pos_base - 1) {
                std::fprintf(stderr, "pending tail position mismatch\n");
                return 2;
            }
            batch_add_embd(batch_tail, pending_tail_hidden.data(), n_embd, pending_tail_pos, true);
            has_pending_tail = false;
        }
        for (int32_t i = 0; i < n_round; ++i) {
            batch_add_embd(batch_tail, draft_hidden.data() + (size_t) i * n_embd, n_embd, pos_base + i, true);
        }
        if (batch_tail.n_tokens > 0 && !tail.decode(batch_tail)) {
            std::fprintf(stderr, "tail verify decode failed\n");
            return 2;
        }

        const int32_t verify_offset = had_pending_tail ? 1 : 0;
        const int32_t n_verify_logits = n_round + verify_offset;
        std::vector<std::vector<float>> verify_logits(n_verify_logits);
        for (int32_t i = 0; i < n_verify_logits; ++i) {
            if (!tail.copy_logits(i, verify_logits[i])) {
                return 2;
            }
        }
        t_tail_verify_ms += elapsed_ms(phase_start);

        ++n_verify;
        n_drafted += n_round;
        int32_t accepted = 0;
        llama_token correction = LLAMA_TOKEN_NULL;
        phase_start = clock::now();
        for (int32_t i = 0; i < n_round; ++i) {
            const std::vector<float> & expected_logits = had_pending_tail || i > 0 ?
                    verify_logits[i + verify_offset - 1] : target_logits;
            const llama_token expected = argmax_token(expected_logits.data(), n_vocab);
            if (draft[i] != expected) {
                correction = expected;
                break;
            }
            print_token(vocab, draft[i]);
            generated.push_back(draft[i]);
            ++accepted;
            ++n_generated;
            target_logits = verify_logits[i + verify_offset];
            if (!opts.ignore_eos && llama_vocab_is_eog(vocab, draft[i])) {
                stop = true;
                break;
            }
        }
        n_accept += accepted;
        t_match_ms += elapsed_ms(phase_start);

        if (stop) {
            continue;
        }

        if (accepted == n_round) {
            if (n_generated >= opts.n_predict) {
                continue;
            }
            ++n_last_completed;

            phase_start = clock::now();
            const llama_token bonus = argmax_token(target_logits.data(), n_vocab);
            const llama_pos bonus_pos = n_prompt + n_generated;
            print_token(vocab, bonus);
            generated.push_back(bonus);
            ++n_generated;
            ++n_bonus;
            if (!opts.ignore_eos && llama_vocab_is_eog(vocab, bonus)) {
                stop = true;
                t_bonus_early_ms += elapsed_ms(phase_start);
                continue;
            }
            if (n_generated >= opts.n_predict) {
                t_bonus_early_ms += elapsed_ms(phase_start);
                continue;
            }

            batch_early.n_tokens = 0;
            batch_add_token(batch_early, bonus, bonus_pos, true);
            if (llama_decode(ctx_early, batch_early) != 0) {
                return 2;
            }
            float * bonus_hidden = llama_get_embeddings_layer_inp(ctx_early, opts.exit_layer);
            pending_tail_hidden.assign(bonus_hidden, bonus_hidden + n_embd);
            pending_tail_pos = bonus_pos;
            has_pending_tail = true;
            if (!copy_logits(ctx_early, -1, n_vocab, early_logits)) {
                return 2;
            }
            t_bonus_early_ms += elapsed_ms(phase_start);
            continue;
        }

        ++n_last_skipped;

        const llama_pos reject_pos = pos_base + accepted;
        phase_start = clock::now();
        if (!llama_memory_seq_rm(llama_get_memory(ctx_early), 0, reject_pos, -1) ||
            !tail.memory_seq_rm(0, reject_pos, -1)) {
            std::fprintf(stderr, "partial KV rollback failed\n");
            return 2;
        }
        t_rollback_ms += elapsed_ms(phase_start);

        phase_start = clock::now();
        print_token(vocab, correction);
        generated.push_back(correction);
        ++n_generated;
        if (!opts.ignore_eos && llama_vocab_is_eog(vocab, correction)) {
            stop = true;
            t_correction_early_ms += elapsed_ms(phase_start);
            continue;
        }
        if (n_generated >= opts.n_predict) {
            t_correction_early_ms += elapsed_ms(phase_start);
            continue;
        }

        batch_early.n_tokens = 0;
        batch_add_token(batch_early, correction, reject_pos, true);
        early_residency_diag(model, opts.exit_layer, "correction.before");
        if (early_diag_call("correction.decode", [&]() { return llama_decode(ctx_early, batch_early); }) != 0) {
            return 2;
        }
        early_residency_diag(model, opts.exit_layer, "correction.after");
        float * correction_hidden = early_diag_call("correction.hidden_sync", [&]() { return llama_get_embeddings_layer_inp(ctx_early, opts.exit_layer); });
        if (!early_diag_call("correction.logits", [&]() { return copy_logits(ctx_early, -1, n_vocab, early_logits); })) {
            return 2;
        }
        early_diag_call("correction.hidden_copy", [&]() {
            pending_tail_hidden.assign(correction_hidden, correction_hidden + n_embd);
            return true;
        });
        pending_tail_pos = reject_pos;
        has_pending_tail = true;
        t_correction_early_ms += elapsed_ms(phase_start);
    }
    }

    const auto time_end = clock::now();
    const double seconds = std::chrono::duration<double>(time_end - time_start).count();
    const double generation_seconds = std::chrono::duration<double>(time_end - generation_start).count();
    const double accept = n_drafted > 0 ? 100.0 * n_accept / n_drafted : 0.0;
    const double window_accept = n_verify > 0 ? 100.0 * n_accept / (n_verify * opts.draft_max) : 0.0;

    std::fprintf(stderr, "\nlayerskip decoded %d tokens in %.3f seconds, speed: %.3f t/s\n", n_generated, seconds, n_generated / seconds);
    std::fprintf(stderr, "layerskip generation %d tokens in %.3f seconds, speed: %.3f t/s\n",
            n_generated, generation_seconds, n_generated / generation_seconds);
    std::fprintf(stderr, "n_drafted = %d\n", n_drafted);
    std::fprintf(stderr, "n_accept  = %d\n", n_accept);
    std::fprintf(stderr, "n_verify  = %d\n", n_verify);
    std::fprintf(stderr, "n_bonus   = %d\n", n_bonus);
    std::fprintf(stderr, "n_last_completed = %d\n", n_last_completed);
    std::fprintf(stderr, "n_last_skipped   = %d\n", n_last_skipped);
    std::fprintf(stderr, "accept = %.3f%%\n", accept);
    std::fprintf(stderr, "window accept = %.3f%%\n", window_accept);
    if (opts.draft_tree) {
        const double leaves_per_verify = n_verify > 0 ? 1.0 + (double) n_tree_branches / n_verify : 0.0;
        std::fprintf(stderr, "tree nodes = %d\n", n_tree_nodes);
        std::fprintf(stderr, "tree branches = %d\n", n_tree_branches);
        std::fprintf(stderr, "tree capacity-skipped branches = %d\n", n_tree_capacity_skipped);
        std::fprintf(stderr, "tree leaves per verify = %.3f\n", leaves_per_verify);
        std::fprintf(stderr, "side branch entries = %d\n", n_side_entries);
        std::fprintf(stderr, "side branch accepted tokens = %d\n", n_side_accept);
        std::fprintf(stderr, "tree tail prefetch calls = %llu\n", (unsigned long long) tail.tree_prefetch_calls);
        std::fprintf(stderr, "tree tail prefetch time = %.3f ms\n", tail.tree_prefetch_time_us / 1000.0);
        std::fprintf(stderr, "tree tail prefetch wait = %.3f ms\n", tail.tree_prefetch_wait_us / 1000.0);
        std::fprintf(stderr, "tree tail prefetch overlap = %.3f ms\n",
                (tail.tree_prefetch_time_us - std::min(tail.tree_prefetch_time_us, tail.tree_prefetch_wait_us)) / 1000.0);
        std::fprintf(stderr, "tree HTP preload calls = %llu\n", (unsigned long long) tail.tree_htp_preload_calls);
        std::fprintf(stderr, "tree HTP preload failures = %llu\n", (unsigned long long) tail.tree_htp_preload_failures);
        std::fprintf(stderr, "tree HTP preload time = %.3f ms\n", tail.tree_htp_preload_time_us / 1000.0);
        std::fprintf(stderr, "tree HTP preload wait = %.3f ms\n", tail.tree_htp_preload_wait_us / 1000.0);
        std::fprintf(stderr, "tree HTP preload overlap = %.3f ms\n",
                (tail.tree_htp_preload_time_us - std::min(tail.tree_htp_preload_time_us, tail.tree_htp_preload_wait_us)) / 1000.0);
    }
    print_token_trace(generated);
    std::fprintf(stderr, "phase prompt early       = %9.3f ms\n", t_prompt_early_ms);
    std::fprintf(stderr, "phase prompt tail        = %9.3f ms\n", t_prompt_tail_ms);
    std::fprintf(stderr, "phase draft build        = %9.3f ms\n", t_draft_build_ms);
    std::fprintf(stderr, "phase tail verify        = %9.3f ms\n", t_tail_verify_ms);
    std::fprintf(stderr, "phase match              = %9.3f ms\n", t_match_ms);
    std::fprintf(stderr, "phase last early         = %9.3f ms\n", t_last_early_ms);
    std::fprintf(stderr, "phase last tail          = %9.3f ms\n", t_last_tail_ms);
    std::fprintf(stderr, "phase bonus early        = %9.3f ms\n", t_bonus_early_ms);
    std::fprintf(stderr, "phase KV rollback        = %9.3f ms\n", t_rollback_ms);
    std::fprintf(stderr, "phase correction early   = %9.3f ms\n", t_correction_early_ms);
    const double measured_generation_ms = t_draft_build_ms + t_tail_verify_ms + t_match_ms +
            t_last_early_ms + t_last_tail_ms + t_bonus_early_ms + t_rollback_ms + t_correction_early_ms;
    std::fprintf(stderr, "phase generation other   = %9.3f ms\n", generation_seconds * 1000.0 - measured_generation_ms);
    if (opts.tail_ffn_stream || opts.tail_layer_stream) {
        std::fprintf(stderr, "tail stream layer loads = %llu\n", (unsigned long long) tail.layer_loads);
        std::fprintf(stderr, opts.tail_layer_stream ? "tail stream weight bytes = %llu\n" : "tail stream FFN bytes   = %llu\n",
                (unsigned long long) tail.stream_bytes);
        std::fprintf(stderr, "tail stream prefetch calls = %llu\n", (unsigned long long) tail.prefetch_calls);
        std::fprintf(stderr, "tail stream prefetch failures = %llu\n", (unsigned long long) tail.prefetch_failures);
        std::fprintf(stderr, "tail stream prefetch time = %.3f ms\n", tail.prefetch_time_us / 1000.0);
        std::fprintf(stderr, "tail stream prefetch wait = %.3f ms\n", tail.prefetch_wait_us / 1000.0);
        std::fprintf(stderr, "tail HTP preload calls = %llu\n", (unsigned long long) tail.htp_preload_calls);
        std::fprintf(stderr, "tail HTP preload failures = %llu\n", (unsigned long long) tail.htp_preload_failures);
        std::fprintf(stderr, "tail HTP preload bytes = %llu\n", (unsigned long long) tail.htp_preload_bytes);
        std::fprintf(stderr, "tail HTP preload time = %.3f ms\n", tail.htp_preload_time_us / 1000.0);
        std::fprintf(stderr, "tail HTP overlap preload calls = %llu\n", (unsigned long long) tail.overlap_htp_preload_calls);
        std::fprintf(stderr, "tail HTP overlap preload failures = %llu\n", (unsigned long long) tail.overlap_htp_preload_failures);
        std::fprintf(stderr, "tail HTP overlap preload time = %.3f ms\n", tail.overlap_htp_preload_time_us / 1000.0);
        std::fprintf(stderr, "tail HTP overlap preload wait = %.3f ms\n", tail.overlap_htp_preload_wait_us / 1000.0);
        std::fprintf(stderr, "tail HTP main prepare time = %.3f ms\n", tail.htp_prepare_time_us / 1000.0);
        std::fprintf(stderr, "tail HTP I/O CPU = %d\n", tail.htp_io_cpu);
        std::fprintf(stderr, "tail HTP I/O affinity failures = %llu\n", (unsigned long long) tail.htp_io_affinity_failures);
        std::fprintf(stderr, "tail HTP direct loads = %llu\n", (unsigned long long) tail.htp_direct_loads);
        std::fprintf(stderr, "tail HTP direct bytes = %llu\n", (unsigned long long) tail.htp_direct_bytes);
        std::fprintf(stderr, "tail HTP compute wait = %.3f ms\n", tail.htp_compute_wait_us / 1000.0);
        std::fprintf(stderr, "tail stream discard failures = %llu\n", (unsigned long long) tail.discard_failures);
        std::fprintf(stderr, "tail stream cache layers = %d\n", opts.tail_stream_cache_layers);
        std::fprintf(stderr, "tail stream cache hits = %llu\n", (unsigned long long) tail.cache_hits);
        std::fprintf(stderr, "tail stream cache bytes = %llu\n", (unsigned long long) tail.cache_bytes);
    }

    llama_batch_free(batch_tail);
    llama_batch_free(batch_early);
    tail.clear();
    llama_free(ctx_early);
    llama_model_free(model);

    return 0;
}
