#pragma once
#include "llama.h"
#include <cstdint>
// Declarations of symbols exported by the frozen 2026-10-18 deployment.
// Layer-weight APIs have C++ linkage; allocation-budget APIs have C linkage.
bool llama_prepare_layer_weights_direct(llama_context *);
int64_t llama_read_layer_weights_direct(llama_context *);
bool llama_commit_layer_weights_direct(llama_context *);
extern "C" {
bool ggml_backend_allocation_budget_set(uint64_t);
uint64_t ggml_backend_allocation_budget_limit();
uint64_t ggml_backend_allocation_budget_used();
uint64_t ggml_backend_allocation_budget_peak();
uint64_t ggml_backend_allocation_budget_failures();
}
