#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>

namespace htp_mix_compact {
// Private Q2/Q3 ABI. Historical high-bit identifiers retain their names in
// the common allocator; kQ4Bytes/Flag now mean the HIGH (3-bit) record only.
static constexpr size_t kHeaderBytes=4096, kQ2Bytes=320, kQ4Bytes=448;
static constexpr uint32_t kTensorTag=0x434d5033u, kQ4Flag=0x80000000u;
#pragma pack(push,1)
struct file_header {
    char magic[8];
    uint32_t version,bits_low,bits_high,group;
    uint64_t k,n,tiles,payload_bytes,descriptor_bytes,data_bytes;
};
#pragma pack(pop)
static_assert(sizeof(file_header)==72,"compact file ABI");
inline size_t descriptor_padded(size_t tiles) { return (tiles*4+127)&~size_t(127); }
inline bool valid_header(const file_header &h,uint64_t k,uint64_t n,uint64_t file_size) {
    if(memcmp(h.magic,"HTPCM23\0",8)||h.version!=1||h.bits_low!=2||h.bits_high!=3||h.group!=32||
       !k||!n||k%32||n%32||h.k!=k||h.n!=n||k/32>UINT64_MAX/(n/32))return false;
    const uint64_t tiles=(k/32)*(n/32);
    // Descriptors use 31-bit payload offsets, and DSP fields use 32-bit sizes.
    if(tiles>UINT32_MAX/448u||tiles>SIZE_MAX/448u)return false;
    const uint64_t desc=descriptor_padded((size_t)tiles);
    return h.tiles==tiles&&h.descriptor_bytes==desc&&h.payload_bytes>=tiles*320&&
      h.payload_bytes<=tiles*448&&h.payload_bytes<kQ4Flag&&
      h.data_bytes==desc+h.payload_bytes&&h.data_bytes<=UINT32_MAX&&
      file_size==kHeaderBytes+h.data_bytes;
}
inline uint32_t descriptor_at(const void *data,size_t tile) {
    uint32_t d;memcpy(&d,(const uint8_t*)data+tile*4,4);return d;
}
inline bool validate_data(const file_header &h,const void *data) {
    uint64_t offset=0;
    const uint8_t *payload=(const uint8_t*)data+h.descriptor_bytes;
    for(size_t t=0;t<h.tiles;++t){
        const uint32_t d=descriptor_at(data,t);const bool q4=(d&kQ4Flag)!=0;
        const size_t bytes=q4?kQ4Bytes:kQ2Bytes;
        if((d&~kQ4Flag)!=offset||offset+bytes>h.payload_bytes)return false;
        if(q4){
            for(size_t s=384;s<448;s+=2){uint16_t v;memcpy(&v,payload+offset+s,2);if((v&0x7c00)==0x7c00)return false;}
        }
        offset+=bytes;
    }
    return offset==h.payload_bytes;
}
}
