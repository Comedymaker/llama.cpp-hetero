// Private HTP mixed Q2/Q4 sidecar ABI.  This is deliberately not a GGUF type.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

namespace htp_mix_g32 {
static constexpr size_t kHeaderBytes = 4096, kMapAlign = 128;
static constexpr size_t kTile = 32, kQ2Bytes = 320, kQ4Bytes = 576;
static constexpr uint16_t kQ2Marker = 0x7e55;

#pragma pack(push, 1)
struct file_header {
    char magic[8];                 // "HTPMIX1\0"
    uint32_t version;              // 1
    uint32_t bits_low, bits_high;  // 2, 4
    uint32_t group;                // 32
    uint64_t k, n, tiles;
    uint64_t payload_bytes, map_bytes;
};
#pragma pack(pop)
static_assert(sizeof(file_header) == 64, "mixed header ABI");

inline size_t round_up(size_t n, size_t a) { return (n + a - 1) / a * a; }
inline size_t map_padded(size_t n) { return round_up(n, kMapAlign); }
inline size_t map_offset() { return kHeaderBytes; }
inline size_t payload_offset(size_t map_bytes) { return kHeaderBytes + map_padded(map_bytes); }
inline bool q4_at(const uint8_t * map, size_t tile) { return (map[tile >> 3] >> (tile & 7)) & 1; }
inline void set_q4(uint8_t * map, size_t tile) { map[tile >> 3] |= uint8_t(1u << (tile & 7)); }

inline bool valid_header(const file_header & h, uint64_t k, uint64_t n, uint64_t file_size) {
    const uint64_t tiles = (k / kTile) * (n / kTile);
    return !memcmp(h.magic, "HTPMIX1\0", 8) && h.version == 1 && h.bits_low == 2 &&
        h.bits_high == 4 && h.group == kTile && h.k == k && h.n == n && h.tiles == tiles &&
        h.map_bytes == (tiles + 7) / 8 && h.payload_bytes >= tiles * kQ2Bytes &&
        file_size == payload_offset((size_t) h.map_bytes) + h.payload_bytes;
}

// Expand a variable Q2 record into its 576-byte physical carrier.  The DSP
// recognizes the marker; bytes 320..573 are intentionally deterministic zero.
inline void decode2physical(const uint8_t in[kQ2Bytes], uint8_t out[kQ4Bytes]) {
    memset(out, 0, kQ4Bytes); memcpy(out, in, kQ2Bytes);
    out[574] = uint8_t(kQ2Marker); out[575] = uint8_t(kQ2Marker >> 8);
}
inline bool q4_scale_finite(const uint8_t tile[kQ4Bytes]) {
    for (size_t i = 512; i < 576; i += 2) { // IEEE fp16 exponent all ones is invalid.
        const uint16_t h = uint16_t(tile[i]) | (uint16_t(tile[i + 1]) << 8);
        if ((h & 0x7c00u) == 0x7c00u) return false;
    }
    return true;
}
// Q4 is already the exact host/DSP tiled order: 512 interleaved nibble bytes,
// followed by 32 fp16 scales.  Keep this copy helper for probes and invariants.
inline void decode4physical(const uint8_t in[kQ4Bytes], uint8_t out[kQ4Bytes]) { memcpy(out, in, kQ4Bytes); }
} // namespace htp_mix_g32
