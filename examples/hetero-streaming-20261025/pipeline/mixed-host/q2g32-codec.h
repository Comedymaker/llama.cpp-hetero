#pragma once

// A deliberately small, host-side 2-bit format.  This is not ggml Q2_K.
// A 32x32 tile is 320 bytes: 256 packed-code bytes followed by 32 fp16
// scales (64 bytes).  Tiles are ordered column-tile (ct) first, then
// k-tile (kt); the source matrix is ordinary row-major [n][k] fp16.

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace q2g32 {

constexpr int kGroup = 32;
constexpr size_t kCodeBytesPerTile = 256;
constexpr size_t kScaleBytesPerTile = 64;
constexpr size_t kTileBytes = kCodeBytesPerTile + kScaleBytesPerTile;

inline uint16_t load_fp16_bits(const uint8_t * p) {
    uint16_t bits;
    std::memcpy(&bits, p, sizeof(bits));
    return bits;
}

inline void store_fp16_bits(uint8_t * p, uint16_t bits) {
    std::memcpy(p, &bits, sizeof(bits));
}

inline void check_shape(const std::vector<uint16_t> & input, int k, int n) {
    if (k <= 0 || n <= 0 || k % kGroup != 0 || n % kGroup != 0) {
        throw std::invalid_argument("q2g32 requires positive k,n that are multiples of 32");
    }
    if (input.size() != static_cast<size_t>(k) * static_cast<size_t>(n)) {
        throw std::invalid_argument("q2g32 input size does not match k*n");
    }
}

inline size_t encoded_size(int k, int n) {
    if (k <= 0 || n <= 0 || k % kGroup != 0 || n % kGroup != 0) {
        throw std::invalid_argument("q2g32 requires positive k,n that are multiples of 32");
    }
    return static_cast<size_t>(k / kGroup) * static_cast<size_t>(n / kGroup) * kTileBytes;
}

// Decode one tile to row-major fp16.  The fp32 product is converted once.
inline void decode_tile(const uint8_t * tile, uint16_t * out_32x32) {
    for (int r = 0; r < kGroup; ++r) {
        const float d = ggml_fp16_to_fp32(load_fp16_bits(tile + kCodeBytesPerTile + 2 * r));
        for (int cp = 0; cp < kGroup / 4; ++cp) {
            const uint8_t packed = tile[cp * kGroup + r];
            for (int lane = 0; lane < 4; ++lane) {
                const int code = (packed >> (2 * lane)) & 3;
                out_32x32[r * kGroup + 4 * cp + lane] =
                    ggml_fp32_to_fp16(d * static_cast<float>(code - 2));
            }
        }
    }
}

inline std::vector<uint16_t> decode_matrix(const std::vector<uint8_t> & encoded, int k, int n) {
    const size_t bytes = encoded_size(k, n);
    if (encoded.size() != bytes) {
        throw std::invalid_argument("q2g32 encoded size does not match k,n");
    }
    std::vector<uint16_t> output(static_cast<size_t>(k) * static_cast<size_t>(n));
    const int kt_count = k / kGroup;
    for (int ct = 0; ct < n / kGroup; ++ct) {
        for (int kt = 0; kt < kt_count; ++kt) {
            const uint8_t * tile = encoded.data() + (static_cast<size_t>(ct) * kt_count + kt) * kTileBytes;
            uint16_t local[kGroup * kGroup];
            decode_tile(tile, local);
            for (int r = 0; r < kGroup; ++r) {
                std::memcpy(output.data() + static_cast<size_t>(ct * kGroup + r) * k + kt * kGroup,
                            local + r * kGroup, kGroup * sizeof(uint16_t));
            }
        }
    }
    return output;
}

inline std::vector<uint8_t> encode_matrix(const std::vector<uint16_t> & input, int k, int n) {
    check_shape(input, k, n);
    for (uint16_t bits : input) {
        if (!std::isfinite(ggml_fp16_to_fp32(bits))) {
            throw std::invalid_argument("q2g32 does not encode non-finite input");
        }
    }

    const int kt_count = k / kGroup;
    std::vector<uint8_t> encoded(encoded_size(k, n), 0);
    for (int ct = 0; ct < n / kGroup; ++ct) {
        for (int kt = 0; kt < kt_count; ++kt) {
            uint8_t * tile = encoded.data() + (static_cast<size_t>(ct) * kt_count + kt) * kTileBytes;
            for (int r = 0; r < kGroup; ++r) {
                const uint16_t * row = input.data() + static_cast<size_t>(ct * kGroup + r) * k + kt * kGroup;
                float signed_absmax = 0.0f;
                float absmax = 0.0f;
                for (int c = 0; c < kGroup; ++c) {
                    const float x = ggml_fp16_to_fp32(row[c]);
                    // Strict comparison makes equal-magnitude orientation deterministic (first wins).
                    if (std::fabs(x) > absmax) {
                        absmax = std::fabs(x);
                        signed_absmax = x;
                    }
                }
                // Preserve canonical +0 for an all-zero group (rather than +0/-2).
                const uint16_t d_bits = ggml_fp32_to_fp16(
                    absmax == 0.0f ? 0.0f : signed_absmax / -2.0f);
                const float d = ggml_fp16_to_fp32(d_bits); // Quantize against the stored scale.
                if (!std::isfinite(d)) {
                    throw std::range_error("q2g32 fp16 scale overflow");
                }
                store_fp16_bits(tile + kCodeBytesPerTile + 2 * r, d_bits);
                for (int c = 0; c < kGroup; ++c) {
                    int code = 2;
                    if (d != 0.0f) {
                        code = static_cast<int>(std::round(ggml_fp16_to_fp32(row[c]) / d)) + 2;
                        code = std::max(0, std::min(3, code));
                    }
                    const int cp = c / 4;
                    const int lane = c % 4;
                    tile[cp * kGroup + r] |= static_cast<uint8_t>(code << (2 * lane));
                }
            }
        }
    }
    return encoded;
}

} // namespace q2g32
