// Include from the copied q2-group32/prepacked-read.h before its Q2 branch.
#pragma once
#include "mix-format.h"
#include "mix-preadv.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cerrno>
#include <vector>
#include <map>
#include <mutex>
#include <string>

#include <dlfcn.h>
struct mixg32_pending { int fd; size_t offset, bytes; std::string name; };
static std::mutex mixg32_mutex;
static std::map<void *, std::vector<mixg32_pending>> mixg32_jobs;

static bool mixg32_pread_exact(int fd, void * dst, size_t bytes, off_t offset) {
    uint8_t * p = static_cast<uint8_t *>(dst);
    while (bytes) {
        const ssize_t got = pread(fd, p, bytes, offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        p += got; bytes -= (size_t) got; offset += got;
    }
    return true;
}

static bool mixg32_present(void * ctx) {
    std::lock_guard<std::mutex> l(mixg32_mutex); return mixg32_jobs.count(ctx) != 0;
}
static bool mixg32_read(void * ctx, const ggml_tensor * src, void * dst) {
    const char * dir = std::getenv("LLAMA_MIX_G32_DIR"); if (!dir) return false;
    const std::string path = std::string(dir) + "/" + src->name + ".mixg32";
    int fd = open(path.c_str(), O_RDONLY); if (fd < 0) return false;
    htp_mix_g32::file_header h{}; struct stat st{};
    bool ok = fstat(fd, &st) == 0 && mixg32_pread_exact(fd, &h, sizeof h, 0) &&
        src->ne[0] % 32 == 0 && src->ne[1] % 32 == 0 &&
        htp_mix_g32::valid_header(h, src->ne[0], src->ne[1], st.st_size);
    const size_t map_n = (size_t) ((src->ne[0] / 32) * (src->ne[1] / 32) + 7) / 8;
    const size_t po = htp_mix_g32::payload_offset(map_n);
    // Capacity persists per reader thread; file contents do not survive a read.
    static thread_local std::vector<uint8_t> map, records;
    const bool direct=std::getenv("LLAMA_MIX_PREADV")&&std::strcmp(std::getenv("LLAMA_MIX_PREADV"),"1")==0;
    map.assign(map_n, 0); records.clear();
    const int64_t before_begin = ggml_time_us();
    const int before_rc = ok ? posix_fadvise(fd, 0, po + (size_t) h.payload_bytes, POSIX_FADV_DONTNEED) : -1;
    const int64_t read_begin = ggml_time_us();
    if (ok) ok = mixg32_pread_exact(fd, map.data(), map_n, htp_mix_g32::map_offset());
    size_t expected_payload = 0;
    if (ok) for (size_t t = 0; t < h.tiles; ++t) expected_payload += htp_mix_g32::q4_at(map.data(), t) ? 576 : 320;
    if (ok) ok = expected_payload == h.payload_bytes;
    mix_direct::stats direct_stats;
    if(ok&&direct)ok=mix_direct::read_into(fd,(uint8_t*)dst,map.data(),h.tiles,po,direct_stats);
    if (ok&&!direct) { records.resize((size_t) h.payload_bytes); ok = mixg32_pread_exact(fd, records.data(), records.size(), po); }
    const int64_t read_end = ggml_time_us(); size_t in = 0, q4 = 0;
    for (size_t t = 0; ok && !direct && t < h.tiles; ++t) {
        const bool hi = htp_mix_g32::q4_at(map.data(), t); const size_t n = hi ? 576 : 320;
        if (in + n > records.size()) { ok = false; break; }
        uint8_t physical[576]; if (hi) { if (!htp_mix_g32::q4_scale_finite(records.data()+in)) { ok=false; break; } htp_mix_g32::decode4physical(records.data()+in, physical); ++q4; }
        else htp_mix_g32::decode2physical(records.data()+in, physical);
        memcpy((uint8_t *) dst + t * 576, physical, 576); in += n;
    }
    if(direct){in=direct_stats.bytes;for(size_t t=0;t<h.tiles;++t)q4+=htp_mix_g32::q4_at(map.data(),t);}
    ok = ok && in == h.payload_bytes;
    const int64_t end = ggml_time_us();
    if (!ok) { close(fd); return false; }
    using register_fn = void (*)(void *, size_t);
    static void * backend_handle = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_NOLOAD);
    static register_fn register_mixed = backend_handle ? (register_fn) dlsym(backend_handle, "ggml_hexagon_mixg32_register") : nullptr;
    if (!register_mixed) { close(fd); LLAMA_LOG_ERROR("mixg32 missing native backend registration\n"); return false; }
    register_mixed(dst, ggml_nbytes(src));
    // Keep the fd pending until compute completes; do not evict source pages early.
    const size_t envelope = po + h.payload_bytes;
    { std::lock_guard<std::mutex> l(mixg32_mutex); mixg32_jobs[ctx].push_back({fd, 0, envelope, src->name}); }
    LLAMA_LOG_INFO("mixg32 read tensor=%s payload=%zu map=%zu header_overhead=%zu q4=%zu before_us=%lld read_s=%.6f scatter_s=%.6f staging_peak=%zu before_rc=%d before_begin_us=%lld begin_us=%lld read_end_us=%lld end_us=%lld route=%s calls=%llu direct_prep_us=%lld direct_read_us=%lld direct_validate_us=%lld payload_copy_bytes=%zu\n", src->name, (size_t)h.payload_bytes, map_n, po, q4, (long long)(read_begin-before_begin), (read_end-read_begin)/1e6, (end-read_end)/1e6, direct?0:records.capacity(), before_rc, (long long)before_begin, (long long)read_begin, (long long)read_end, (long long)end,direct?"preadv-to-HTP":"staging-copy",(unsigned long long)direct_stats.calls,(long long)direct_stats.prepare_us,(long long)direct_stats.read_us,(long long)direct_stats.validate_us,direct?0:(size_t)h.tiles*576);
    return true;
}
static void mixg32_after_sync(void * ctx) {
    std::vector<mixg32_pending> jobs; { std::lock_guard<std::mutex> l(mixg32_mutex); auto it=mixg32_jobs.find(ctx); if(it==mixg32_jobs.end()) return; jobs.swap(it->second); mixg32_jobs.erase(it); }
    const int64_t sync = ggml_time_us();
    for (auto & j : jobs) fp16_evictions().submit(j.fd, j.offset, j.bytes, j.name.c_str());
    LLAMA_LOG_INFO("mixg32 after-compute eviction ctx=%p tensors=%zu sync_us=%lld enqueue_end_us=%lld\n", ctx, jobs.size(), (long long)sync, (long long)ggml_time_us());
}
