// Standalone integration control for the Early8 CPU <-> HTP state bridge.
// It intentionally creates no tail context and is not a planner benchmark.
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

static std::vector<uint8_t> control_eog_mask;
static void mask_eog_logits(std::vector<float> & logits) {
    for (size_t i = 0; i < logits.size() && i < control_eog_mask.size(); ++i) {
        if (control_eog_mask[i]) logits[i] = -INFINITY;
    }
}

#include "batch-draft-device.h"

static void add_token(llama_batch & batch, llama_token token, llama_pos pos, llama_seq_id seq) {
    const int row = batch.n_tokens++;
    batch.token[row] = token;
    batch.pos[row] = pos;
    batch.n_seq_id[row] = 1;
    batch.seq_id[row][0] = seq;
    batch.logits[row] = true;
}

static bool finite(const std::vector<float> & values) {
    for (float value : values) if (!std::isfinite(value)) return false;
    return true;
}

// Bridge logits have the planner's EOG policy applied, so -inf is expected
// precisely for EOG entries; all model-produced (non-EOG) values stay finite.
static bool masked_logits_valid(const std::vector<float> & values) {
    for (size_t i = 0; i < values.size(); ++i) {
        if (i < control_eog_mask.size() && control_eog_mask[i]) {
            if (values[i] != -INFINITY) return false;
        } else if (!std::isfinite(values[i])) return false;
    }
    return true;
}

static bool state_get(llama_context * ctx, std::vector<uint8_t> & state) {
    state.resize(llama_state_get_size(ctx));
    return !state.empty() && llama_state_get_data(ctx, state.data(), state.size()) == state.size();
}

static bool state_set(llama_context * ctx, const std::vector<uint8_t> & state) {
    return !state.empty() && llama_state_set_data(ctx, state.data(), state.size()) == state.size();
}

static int top1(const std::vector<float> & values) {
    return int(std::max_element(values.begin(), values.end()) - values.begin());
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL PROMPT_FILE\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    std::ifstream prompt_file(argv[2]);
    const std::string prompt((std::istreambuf_iterator<char>(prompt_file)), {});
    if (!prompt_file || prompt.empty()) {
        std::fprintf(stderr, "bridge-control: cannot read prompt\n");
        return 2;
    }

    int result = 1;
    llama_backend_init();
    llama_model * model = nullptr;
    llama_context * cpu = nullptr;
    llama_batch prefill{};
    llama_batch htp_batch{};
    llama_batch continuation{};
    llama_batch cpu_batch{};
    bool prefill_owned = false, htp_batch_owned = false, continuation_owned = false, cpu_batch_owned = false;
    do {
        ggml_backend_dev_t cpu_device = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
        if (!cpu_device) { std::fprintf(stderr, "bridge-control: CPU backend unavailable\n"); break; }
        ggml_backend_dev_t devices[] = {cpu_device, nullptr};
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        mp.tensor_read_lazy = LLAMA_TENSOR_READ_LAZY_ON;
        mp.devices = devices;
        const llama_model_tensor_buft_override overrides[] = {
            {"^blk\\.[0-9]+\\.", ggml_backend_cpu_buffer_type()},
            {"^(token_embd|output|output_norm)\\.(weight|bias)$", ggml_backend_cpu_buffer_type()},
            {nullptr, nullptr},
        };
        mp.tensor_buft_overrides = overrides;
        model = llama_model_load_from_file(model_path.c_str(), mp);
        if (!model) { std::fprintf(stderr, "bridge-control: CPU model load failed\n"); break; }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        const int n_embd = llama_model_n_embd(model);
        std::vector<llama_token> prompt_tokens(prompt.size() + 8);
        const int n_prompt = llama_tokenize(vocab, prompt.data(), int(prompt.size()), prompt_tokens.data(), int(prompt_tokens.size()), true, false);
        if (n_prompt != 133) { std::fprintf(stderr, "bridge-control: expected HumanEval000 133 tokens, got %d\n", n_prompt); break; }
        prompt_tokens.resize(n_prompt);
        control_eog_mask.assign(n_vocab, 0);
        for (int token = 0; token < n_vocab; ++token) control_eog_mask[token] = llama_vocab_is_eog(vocab, token);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 2048; cp.n_batch = 512; cp.n_ubatch = 128; cp.n_seq_max = 64;
        cp.kv_unified = true; cp.n_threads = 4; cp.n_threads_batch = 4;
        cp.offload_kqv = false; cp.n_layer_start = 0; cp.n_layer_end = 8;
        cpu = llama_init_from_model(model, cp);
        if (!cpu) { std::fprintf(stderr, "bridge-control: CPU Early8 context failed\n"); break; }
        llama_set_embeddings_layer_inp(cpu, 8, true);

        prefill = llama_batch_init(n_prompt, 0, 1); prefill_owned = true;
        for (int i = 0; i < n_prompt; ++i) {
            add_token(prefill, prompt_tokens[i], i, 0);
            prefill.logits[i] = i == n_prompt - 1;
        }
        if (llama_decode(cpu, prefill) != 0) { std::fprintf(stderr, "bridge-control: prefill failed\n"); break; }
        float * root_logits_raw = llama_get_logits_ith(cpu, -1);
        if (!root_logits_raw) { std::fprintf(stderr, "bridge-control: root logits unavailable\n"); break; }
        std::vector<float> root_logits(root_logits_raw, root_logits_raw + n_vocab);
        if (!finite(root_logits)) { std::fprintf(stderr, "bridge-control: root logits non-finite\n"); break; }
        mask_eog_logits(root_logits);
        std::vector<int> order(n_vocab); std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + 8, order.end(), [&](int a, int b) {
            return root_logits[a] == root_logits[b] ? a < b : root_logits[a] > root_logits[b];
        });
        std::vector<llama_token> candidates(order.begin(), order.begin() + 8);
        if (std::any_of(candidates.begin(), candidates.end(), [&](llama_token t) { return llama_vocab_is_eog(vocab, t); })) {
            std::fprintf(stderr, "bridge-control: candidate selection included EOG\n"); break;
        }
        std::vector<uint8_t> root_state;
        if (!state_get(cpu, root_state)) { std::fprintf(stderr, "bridge-control: root state export failed\n"); break; }

        bool branch_positions_ok = false, continuation_ok = false, root_restored_exact = false;
        int top1_agree = 0;
        double maxabs = 0, rms = 0, transfer_ms = 0, forward_ms = 0;
        size_t bridge_state_bytes = 0, bridge_state_bytes_max = 0;
        {
            batch_draft_htp bridge;
            if (!bridge.init(model_path, cpu, 8, 2048, 512, 128, 64, 4)) break;
            llama_memory_t memory = llama_get_memory(cpu);
            for (int seq = 1; seq <= 8; ++seq) llama_memory_seq_cp(memory, 0, seq, 0, n_prompt);
            htp_batch = llama_batch_init(8, 0, 8); htp_batch_owned = true;
            for (int row = 0; row < 8; ++row) add_token(htp_batch, candidates[row], n_prompt, row + 1);
            std::vector<std::vector<float>> htp_logits, htp_hidden;
            if (!bridge.decode(cpu, htp_batch, n_vocab, n_embd, htp_logits, htp_hidden, transfer_ms, forward_ms)) break;
            if (htp_logits.size() != 8 || htp_hidden.size() != 8 ||
                    !std::all_of(htp_logits.begin(), htp_logits.end(), masked_logits_valid) ||
                    !std::all_of(htp_hidden.begin(), htp_hidden.end(), finite)) {
                std::fprintf(stderr, "bridge-control: invalid HTP output shape or values\n"); break;
            }
            branch_positions_ok = llama_memory_seq_pos_max(memory, 0) == n_prompt - 1;
            for (int seq = 1; seq <= 8; ++seq) branch_positions_ok = branch_positions_ok && llama_memory_seq_pos_max(memory, seq) == n_prompt;
            if (!branch_positions_ok) { std::fprintf(stderr, "bridge-control: returned KV positions are wrong\n"); break; }

            continuation = llama_batch_init(1, 0, 1); continuation_owned = true;
            add_token(continuation, top1(htp_logits[0]), n_prompt + 1, 1);
            if (llama_decode(cpu, continuation) != 0 || llama_memory_seq_pos_max(memory, 1) != n_prompt + 1) {
                std::fprintf(stderr, "bridge-control: CPU continuation decode/position failed\n"); break;
            }
            float * continuation_logits = llama_get_logits_ith(cpu, 0);
            if (!continuation_logits || !finite(std::vector<float>(continuation_logits, continuation_logits + n_vocab))) {
                std::fprintf(stderr, "bridge-control: CPU continuation logits non-finite\n"); break;
            }
            continuation_ok = true;

            if (!state_set(cpu, root_state)) { std::fprintf(stderr, "bridge-control: root state restore failed\n"); break; }
            std::vector<uint8_t> restored;
            if (!state_get(cpu, restored) || restored.size() != root_state.size() ||
                    std::memcmp(restored.data(), root_state.data(), root_state.size()) != 0) {
                std::fprintf(stderr, "bridge-control: root state did not restore byte-exactly\n"); break;
            }
            root_restored_exact = true;

            for (int seq = 1; seq <= 8; ++seq) llama_memory_seq_cp(llama_get_memory(cpu), 0, seq, 0, n_prompt);
            cpu_batch = llama_batch_init(8, 0, 8); cpu_batch_owned = true;
            for (int row = 0; row < 8; ++row) add_token(cpu_batch, candidates[row], n_prompt, row + 1);
            if (llama_decode(cpu, cpu_batch) != 0) { std::fprintf(stderr, "bridge-control: CPU comparison batch failed\n"); break; }
            double ss = 0; size_t count = 0;
            bool comparison_ok = true;
            for (int row = 0; row < 8; ++row) {
                float * raw = llama_get_logits_ith(cpu, row);
                if (!raw) { std::fprintf(stderr, "bridge-control: CPU comparison logits unavailable\n"); comparison_ok = false; break; }
                std::vector<float> cpu_logits(raw, raw + n_vocab);
                if (!finite(cpu_logits)) { std::fprintf(stderr, "bridge-control: CPU comparison logits non-finite\n"); comparison_ok = false; break; }
                mask_eog_logits(cpu_logits);
                top1_agree += top1(cpu_logits) == top1(htp_logits[row]);
                for (int token = 0; token < n_vocab; ++token) if (std::isfinite(cpu_logits[token]) && std::isfinite(htp_logits[row][token])) {
                    const double delta = double(cpu_logits[token]) - htp_logits[row][token];
                    maxabs = std::max(maxabs, std::abs(delta)); ss += delta * delta; ++count;
                }
            }
            if (!comparison_ok || !count) { std::fprintf(stderr, "bridge-control: no complete finite comparison logits\n"); break; }
            rms = std::sqrt(ss / count);
            bridge_state_bytes = bridge.state_bytes;
            bridge_state_bytes_max = bridge.state_bytes_max;
            std::printf("{\"ok\":true,\"state_bytes\":%zu,\"state_bytes_max\":%zu,\"transfer_ms\":%.3f,\"forward_ms\":%.3f,\"root_restored_exact\":true,\"cpu_continuation_passed\":true,\"branch_positions_passed\":true,\"top1_agreement\":%d,\"rows\":8,\"logits_maxabs\":%.8g,\"logits_rms\":%.8g}\n",
                    bridge_state_bytes, bridge_state_bytes_max, transfer_ms, forward_ms, top1_agree, maxabs, rms);
            result = 0;
        }
    } while (false);
    if (cpu_batch_owned) llama_batch_free(cpu_batch);
    if (continuation_owned) llama_batch_free(continuation);
    if (htp_batch_owned) llama_batch_free(htp_batch);
    if (prefill_owned) llama_batch_free(prefill);
    if (cpu) llama_free(cpu);
    if (model) llama_model_free(model);
    llama_backend_free();
    return result;
}
