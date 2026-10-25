#pragma once

// A deliberately boring CPU <-> HTP bridge for the early-only model.  In
// particular, this moves the *whole* context state: copying one prefix per
// frontier sequence loses the aliases created by llama_memory_seq_cp and can
// overrun the unified KV capacity.
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// queue.cpp owns the EOG policy.  Its existing definition may appear either
// before or after this header is included in that translation unit.
static void mask_eog_logits(std::vector<float> & logits);

class batch_draft_htp {
public:
    double transfer_prior_ms = 0;
    size_t state_bytes = 0;
    size_t state_bytes_max = 0;
    double cpu_to_htp_ms = 0;
    double htp_to_cpu_ms = 0;
    inline static std::atomic<unsigned long long> global_handoffs{0};

    batch_draft_htp() = default;
    batch_draft_htp(const batch_draft_htp &) = delete;
    batch_draft_htp & operator=(const batch_draft_htp &) = delete;

    ~batch_draft_htp() {
        if (htp_ctx_) llama_free(htp_ctx_);
        if (htp_model_) llama_model_free(htp_model_);
    }

    bool init(const std::string & path, llama_context * cpu, int exitLayer, int nctx,
            int nbatch, int nubatch, int nseq, int threads) {
        if (!cpu || path.empty() || exitLayer <= 0 || nctx <= 0 || nbatch <= 0 ||
                nubatch <= 0 || nseq <= 0 || threads <= 0) {
            return fail("invalid bridge initialization arguments");
        }
        if (htp_ctx_ || htp_model_) return fail("bridge initialized twice");

        // The tail-only prepack sidecar is not valid for Early tensors.  The
        // model loader therefore uses its normal set_tensor packing exactly
        // once while this scope temporarily hides the sidecar setting.
        prepack_env_guard prepack_guard;
        ggml_backend_load_all();
        ggml_backend_dev_t htp = ggml_backend_dev_by_name("HTP0");
        if (!htp) return fail("HTP0 backend is unavailable");

        ggml_backend_dev_t devices[] = { htp, nullptr };
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        mp.tensor_read_lazy = LLAMA_TENSOR_READ_LAZY_ON;
        mp.devices = devices;

        // Keep the regex order identical to the isolated Early8 probe: the
        // first match sends L0..exitLayer-1 to HTP and the later rules pin
        // every other block plus embedding/head to CPU.
        const std::string early = "^blk\\.(0|1|2|3|4|5|6|7)\\.";
        if (exitLayer != 8) {
            return fail("only the measured Early8 (exit layer 8) placement is supported");
        }
        std::vector<llama_model_tensor_buft_override> overrides;
        overrides.push_back({early.c_str(), ggml_backend_dev_buffer_type(htp)});
        overrides.push_back({"^blk\\.[0-9]+\\.", ggml_backend_cpu_buffer_type()});
        overrides.push_back({"^(token_embd|output|output_norm)\\.(weight|bias)$", ggml_backend_cpu_buffer_type()});
        overrides.push_back({nullptr, nullptr});
        mp.tensor_buft_overrides = overrides.data();

        htp_model_ = llama_model_load_from_file(path.c_str(), mp);
        if (!htp_model_) return fail("HTP Early8 model load failed");

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = nctx;
        cp.n_batch = nbatch;
        cp.n_ubatch = nubatch;
        cp.n_seq_max = nseq;
        cp.kv_unified = true;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.offload_kqv = false;
        cp.n_layer_start = 0;
        cp.n_layer_end = exitLayer;
        htp_ctx_ = llama_init_from_model(htp_model_, cp);
        if (!htp_ctx_) return fail("HTP Early8 context initialization failed");
        llama_set_embeddings_layer_inp(htp_ctx_, exitLayer, true);
        exit_layer_ = exitLayer;
        prepack_guard.restore();

        // Validate the actual, currently-prefilled CPU root before admitting
        // the device to the planner.  State APIs synchronize each context.
        const auto start = clock::now();
        if (!get_state(cpu, cpu_state_, "calibration CPU serialize") ||
                !set_state(htp_ctx_, cpu_state_, "calibration HTP restore") ||
                !get_state(htp_ctx_, htp_state_, "calibration HTP serialize") ||
                !set_state(cpu, htp_state_, "calibration CPU restore")) {
            return false;
        }
        // This timing is the real CPU->HTP->CPU handoff.  The comparison
        // serialization below is intentionally outside the planner prior.
        transfer_prior_ms = milliseconds(start);
        std::vector<uint8_t> after;
        if (!get_state(cpu, after, "calibration CPU verification serialize")) return false;
        if (cpu_state_.size() != after.size() ||
                std::memcmp(cpu_state_.data(), after.data(), cpu_state_.size()) != 0) {
            return fail("full-state roundtrip changed CPU bytes; refusing unsafe handoff");
        }
        ready_ = true;
        std::fprintf(stderr, "batch-bridge: ready Early8 HTP full-state=%zu bytes calibration=%.3f ms nctx=%d nseq=%d\n",
                state_bytes, transfer_prior_ms, nctx, nseq);
        return true;
    }

    bool decode(llama_context * cpu, const llama_batch & batch, int nv, int ne,
            std::vector<std::vector<float>> & logits, std::vector<std::vector<float>> & hidden,
            double & transfer_ms, double & forward_ms) {
        transfer_ms = 0;
        forward_ms = 0;
        if (!ready_ || !cpu || !htp_ctx_ || nv <= 0 || ne <= 0 || batch.n_tokens <= 0) {
            return fail("decode called without a ready bridge or with invalid shape");
        }
        for (int row = 0; row < batch.n_tokens; ++row) {
            if (!batch.logits[row]) return fail("bridge batches must request an output for every row");
        }
        const auto transfer_start = clock::now();
        if (!get_state(cpu, cpu_state_, "CPU->HTP serialize") ||
                !set_state(htp_ctx_, cpu_state_, "CPU->HTP restore")) return false;
        cpu_to_htp_ms = milliseconds(transfer_start);
        transfer_ms += cpu_to_htp_ms;

        const auto forward_start = clock::now();
        if (llama_decode(htp_ctx_, batch) != 0) {
            return fail("HTP Early8 decode failed (CPU state was left unchanged)");
        }
        logits.assign(batch.n_tokens, {});
        hidden.assign(batch.n_tokens, {});
        const float * all_hidden = llama_get_embeddings_layer_inp(htp_ctx_, exit_layer());
        if (!all_hidden) return fail("HTP Early8 hidden output is unavailable");
        for (int row = 0; row < batch.n_tokens; ++row) {
            const float * row_logits = llama_get_logits_ith(htp_ctx_, row);
            if (!row_logits) return fail("HTP Early8 logits output is unavailable");
            logits[row].assign(row_logits, row_logits + nv);
            if (!finite(logits[row])) return fail("HTP Early8 logits contain non-finite values before EOG mask");
            mask_eog_logits(logits[row]);
            hidden[row].assign(all_hidden + size_t(row) * ne, all_hidden + size_t(row + 1) * ne);
            if (!finite(hidden[row])) return fail("HTP Early8 hidden output contains non-finite values");
        }
        forward_ms = milliseconds(forward_start);

        const auto return_start = clock::now();
        if (!get_state(htp_ctx_, htp_state_, "HTP->CPU serialize") ||
                !set_state(cpu, htp_state_, "HTP->CPU restore")) return false;
        htp_to_cpu_ms = milliseconds(return_start);
        transfer_ms += htp_to_cpu_ms;
        transfer_prior_ms = transfer_prior_ms > 0 ? 0.8 * transfer_prior_ms + 0.2 * transfer_ms : transfer_ms;
        const unsigned long long handoff = ++global_handoffs;
        std::fprintf(stderr, "batch-bridge: handoff=%llu rows=%d CPU->HTP=%.3f HTP->CPU=%.3f total=%.3f ms forward=%.3f ms state=%zu max=%zu ema=%.3f ms\n",
                handoff, batch.n_tokens, cpu_to_htp_ms, htp_to_cpu_ms, transfer_ms, forward_ms,
                state_bytes, state_bytes_max, transfer_prior_ms);
        return true;
    }

private:
    using clock = std::chrono::steady_clock;
    llama_model * htp_model_ = nullptr;
    llama_context * htp_ctx_ = nullptr;
    int exit_layer_ = 8;
    std::vector<uint8_t> cpu_state_;
    std::vector<uint8_t> htp_state_;
    bool ready_ = false;

    int exit_layer() const { return exit_layer_; }
    static double milliseconds(clock::time_point start) {
        return std::chrono::duration<double, std::milli>(clock::now() - start).count();
    }
    bool fail(const char * message) const {
        std::fprintf(stderr, "batch-bridge: ERROR: %s\n", message);
        return false;
    }
    bool get_state(llama_context * ctx, std::vector<uint8_t> & out, const char * phase) {
        const size_t expected = llama_state_get_size(ctx);
        if (expected == 0) {
            std::fprintf(stderr, "batch-bridge: ERROR: %s reported an empty state\n", phase);
            return false;
        }
        out.resize(expected);
        const size_t used = llama_state_get_data(ctx, out.data(), out.size());
        if (used != out.size()) {
            std::fprintf(stderr, "batch-bridge: ERROR: %s consumed %zu bytes, expected %zu\n", phase, used, out.size());
            return false;
        }
        state_bytes = used;
        state_bytes_max = std::max(state_bytes_max, used);
        return true;
    }
    bool set_state(llama_context * ctx, const std::vector<uint8_t> & in, const char * phase) const {
        const size_t used = llama_state_set_data(ctx, in.data(), in.size());
        if (used != in.size()) {
            std::fprintf(stderr, "batch-bridge: ERROR: %s consumed %zu bytes, expected %zu\n", phase, used, in.size());
            return false;
        }
        return true;
    }
    static bool finite(const std::vector<float> & values) {
        for (float value : values) if (!std::isfinite(value)) return false;
        return true;
    }

    class prepack_env_guard {
    public:
        prepack_env_guard() {
            const char * current = std::getenv("LLAMA_Q4_PREPACK_DIR");
            had_value_ = current != nullptr;
            if (had_value_) value_ = current;
#if defined(_WIN32)
            _putenv_s("LLAMA_Q4_PREPACK_DIR", "");
#else
            unsetenv("LLAMA_Q4_PREPACK_DIR");
#endif
        }
        ~prepack_env_guard() {
            restore();
        }
        void restore() {
            if (!active_) return;
#if defined(_WIN32)
            _putenv_s("LLAMA_Q4_PREPACK_DIR", had_value_ ? value_.c_str() : "");
#else
            if (had_value_) setenv("LLAMA_Q4_PREPACK_DIR", value_.c_str(), 1);
            else unsetenv("LLAMA_Q4_PREPACK_DIR");
#endif
            active_ = false;
        }
    private:
        bool had_value_ = false;
        bool active_ = true;
        std::string value_;
    };
};
