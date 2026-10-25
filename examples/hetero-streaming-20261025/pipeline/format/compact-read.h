#pragma once
#include "compact-format.h"
// Included after mix-read.h: reuse exact-pread and post-compute eviction ownership.
static bool mixcompact_read(void *ctx,const ggml_tensor *src,void *dst) {
    const char *dir=std::getenv("LLAMA_MIX_COMPACT_DIR");if(!dir)return false;
    const std::string path=std::string(dir)+"/"+src->name+".mixcompact";
    int fd=open(path.c_str(),O_RDONLY);if(fd<0)return false;
    htp_mix_compact::file_header h{};struct stat st{};
    bool ok=fstat(fd,&st)==0&&mixg32_pread_exact(fd,&h,sizeof h,0)&&
        htp_mix_compact::valid_header(h,src->ne[0],src->ne[1],st.st_size);
    const int64_t before=ggml_time_us();
    const int before_rc=ok?posix_fadvise(fd,0,st.st_size,POSIX_FADV_DONTNEED):-1;
    const int64_t begin=ggml_time_us();
    if(ok)ok=mixg32_pread_exact(fd,dst,(size_t)h.data_bytes,htp_mix_compact::kHeaderBytes);
    const int64_t read_end=ggml_time_us();
    if(ok)ok=htp_mix_compact::validate_data(h,dst);
    using register_fn=void(*)(void*,size_t);
    static void *handle=dlopen("libggml-hexagon.so",RTLD_NOW|RTLD_NOLOAD);
    static register_fn reg=handle?(register_fn)dlsym(handle,"ggml_hexagon_mixcompact_register"):nullptr;
    if(!ok||!reg){close(fd);LLAMA_LOG_ERROR("mixcompact invalid load/registration tensor=%s\n",src->name);return false;}
    reg(dst,(size_t)h.data_bytes);
    {std::lock_guard<std::mutex> l(mixg32_mutex);mixg32_jobs[ctx].push_back({fd,0,(size_t)st.st_size,src->name});}
    LLAMA_LOG_INFO("mixcompact read tensor=%s tiles=%llu payload=%llu descriptors=%llu shared_bytes=%llu old_carrier_bytes=%llu before_us=%lld read_us=%lld validate_us=%lld before_rc=%d begin_us=%lld read_end_us=%lld end_us=%lld staging=0 payload_copy_bytes=0 offchip_padding=0\n",
      src->name,(unsigned long long)h.tiles,(unsigned long long)h.payload_bytes,(unsigned long long)h.descriptor_bytes,
      (unsigned long long)h.data_bytes,(unsigned long long)h.tiles*576,(long long)(begin-before),
      (long long)(read_end-begin),(long long)(ggml_time_us()-read_end),before_rc,(long long)begin,(long long)read_end,(long long)ggml_time_us());
    return true;
}
