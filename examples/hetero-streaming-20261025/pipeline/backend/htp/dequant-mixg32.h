// Private mixed Q2/Q3: compact DRAM records320/448B, VTCM stride640B.
// A marker0x7e55 at574 selects Q2; explicit zero selects Q3. split-dma.h
// resets the discriminator for BOTH kinds before loading each compact run.
// Keep the per-format SIMD decoders authoritative: this wrapper only selects
// one and adjusts their expected source stride for the current tile.
static void dequantize_tiled_weight_to_fp16_task_mix_g32(
        const tiled_dequantize_state_t *state, uint32_t start_tile, uint32_t end_tile) {
    for (uint32_t t = start_tile; t < end_tile; ++t) {
        const uint8_t *p = state->src + (size_t)t * HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q4_0;
        tiled_dequantize_state_t one = *state;
        if (p[574] == 0x55 && p[575] == 0x7e) {
            // Existing Q2 task indexes src as t*384.  Bias its base so its
            // original t,t+1 task contract still resolves to this 640 tile.
            one.aligned_tile_size = 384;
            one.tile_size = 320;
            one.src = state->src + (size_t)t * (640 - 384);
            dequantize_tiled_weight_to_fp16_task_q2_g32(&one, t, t + 1);
        } else {
            one.aligned_tile_size = 512;
            one.tile_size = 448;
            one.src = state->src + (size_t)t * (640 - 512);
            dequantize_tiled_weight_to_fp16_task_q3_g32(&one, t, t + 1);
        }
    }
}
