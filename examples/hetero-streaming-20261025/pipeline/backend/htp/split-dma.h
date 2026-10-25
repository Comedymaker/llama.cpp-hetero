// Matched private host/DSP ABI: reserved marker 0x51535031 means Q4 codes
// followed by all scales in shared DRAM. Preserve existing VTCM/dequant math.
static void q4_split_dma_push(struct htp_context*ctx,void*raw,const uint8_t*weight,
 size_t first_tile,size_t count,size_t total_tiles,uint32_t dst_stride){
 dma_queue_push(ctx->dma[0],dma_make_ptr(raw,weight+first_tile*512),dst_stride,512,512,count);
 dma_queue_push(ctx->dma[0],dma_make_ptr((uint8_t*)raw+512,weight+total_tiles*512+first_tile*64),dst_stride,64,64,count);
}
static void *q4_split_dma_pop(struct htp_context*ctx){
 void*p=dma_queue_pop(ctx->dma[0]).dst;dma_queue_pop(ctx->dma[0]);return p;
}

// Private Q2/Q3 compact DRAM ABI (reserved == 0x434d5033):
//   u32 descriptor[total_tiles], rounded up to 128 bytes, then variable
//   records.  A descriptor's high bit selects Q3 (448 bytes); the low bits
//   are the exact byte offset from payload_base.  VTCM deliberately keeps
//   the ordinary 640-byte MIX carrier stride so the existing dequantizer can
//   remain completely unchanged.
#define COMPACT_MIX_G32_MARKER 0x434d5033u
#define COMPACT_MIX_Q2_BYTES   320u
#define COMPACT_MIX_Q4_BYTES   448u
#define COMPACT_MIX_VTCM_STRIDE 640u

static inline size_t compact_mix_payload_base(size_t total_tiles) {
    return (total_tiles * sizeof(uint32_t) + 127u) & ~(size_t)127u;
}

// Hardware DMA contains only real compact runs. This private FIFO tracks the
// logical HMX chunks and their still-enqueued real runs. HMX-2D has at most
// two outstanding weight chunks.
typedef struct {
    struct {
        void *raw;
        uint32_t pending_runs;
    } chunk[2];
    uint32_t head;
    uint32_t count;
} compact_mix_dma_state;

static inline bool compact_mix_dma_retire_one(struct htp_context *ctx,
                                               compact_mix_dma_state *state) {
    // Runs are pushed chunk-by-chunk, so the first logical chunk with a
    // pending run owns the real DMA FIFO head.
    for (uint32_t i = 0; i < state->count; ++i) {
        const uint32_t idx = (state->head + i) & 1u;
        if (state->chunk[idx].pending_runs) {
            dma_queue_pop(ctx->dma[0]);
            --state->chunk[idx].pending_runs;
            return true;
        }
    }
    FARF(ERROR, "compact DMA queue ownership lost");
    return false;
}

static void compact_mix_dma_push(struct htp_context *ctx, compact_mix_dma_state *state, void *raw,
                                 const uint8_t *weight, size_t first_tile,
                                 size_t count, size_t total_tiles,
                                 size_t *payload_bytes) {
    const uint32_t *desc = (const uint32_t *) weight;
    const uint8_t *payload = weight + compact_mix_payload_base(total_tiles);
    uint8_t *dst = (uint8_t *) raw;
    size_t tile = 0;
    if (state->count >= 2) {
        FARF(ERROR, "compact DMA logical queue overflow");
        return;
    }
    const uint32_t chunk_idx = (state->head + state->count) & 1u;
    state->chunk[chunk_idx].raw = raw;
    state->chunk[chunk_idx].pending_runs = 0;
    ++state->count;

    // Reset the discriminator for BOTH short record types; neither overwrites
    // bytes574..575. Reused VTCM must not keep the previous tile's format.
    for (; tile < count; ++tile) {
        const bool low=(desc[first_tile+tile]&0x80000000u)==0;
        // BOTH records are shorter than574. Explicitly reset the discriminator
        // for Q3 too, or a reused VTCM slot can retain an old Q2 marker.
        dst[tile * COMPACT_MIX_VTCM_STRIDE + 574] = low?0x55:0;
        dst[tile * COMPACT_MIX_VTCM_STRIDE + 575] = low?0x7e:0;
    }

    tile = 0;
    while (tile < count) {
        const uint32_t first = desc[first_tile + tile];
        const bool is_q4 = (first & 0x80000000u) != 0;
        const size_t width = is_q4 ? COMPACT_MIX_Q4_BYTES : COMPACT_MIX_Q2_BYTES;
        const size_t offset = (size_t)(first & 0x7fffffffu);
        size_t run = 1;

        // DMA only truly contiguous records of the same physical type.
        while (tile + run < count) {
            const uint32_t next = desc[first_tile + tile + run];
            if (((next & 0x80000000u) != 0) != is_q4 ||
                (size_t)(next & 0x7fffffffu) != offset + run * width) break;
            ++run;
        }

        while (!dma_queue_push(ctx->dma[0],
                               dma_make_ptr(dst + tile * COMPACT_MIX_VTCM_STRIDE,
                                            payload + offset),
                               COMPACT_MIX_VTCM_STRIDE, width, width, run)) {
            // Under ring pressure retire one earlier real DMA run.  Never
            // remove a logical HMX chunk merely to make DMA descriptor room.
            if (!compact_mix_dma_retire_one(ctx, state)) return;
        }
        ++state->chunk[chunk_idx].pending_runs;
        *payload_bytes += run * width;
        tile += run;
    }
}

static void *compact_mix_dma_pop(struct htp_context *ctx, compact_mix_dma_state *state) {
    if (state->count == 0) return NULL;
    const uint32_t chunk_idx = state->head;
    while (state->chunk[chunk_idx].pending_runs) {
        if (!compact_mix_dma_retire_one(ctx, state)) return NULL;
    }
    void *raw = state->chunk[chunk_idx].raw;
    state->head = (state->head + 1u) & 1u;
    --state->count;
    return raw;
}
