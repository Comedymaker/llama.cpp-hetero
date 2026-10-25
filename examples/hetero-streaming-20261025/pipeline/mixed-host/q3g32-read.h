// Private sidecar format, never labelled standard GGUF Q4 or Q2_K.
#include <dlfcn.h>
static bool q3g32_register_storage(void*data,size_t bytes){
 using fn_t=void(*)(void*,size_t);
 static void*handle=dlopen("libggml-hexagon.so",RTLD_NOW|RTLD_NOLOAD);
 static fn_t fn=handle?reinterpret_cast<fn_t>(dlsym(handle,"ggml_hexagon_q3g32_register")):nullptr;
 if(!fn)return false;fn(data,bytes);return true;
}
struct q3g32_file_header {char magic[8];uint32_t version,bits,group,tile_bytes;uint64_t k,n,bytes;};
struct q3g32_pending {int fd;size_t bytes;std::string name;};
static std::mutex q3g32_mutex;
static std::map<void*,std::vector<q3g32_pending>>q3g32_jobs;
static bool q3g32_read(void*ctx,const ggml_tensor*src,void*dst){
 const char*dir=std::getenv("LLAMA_Q3_GROUP32_DIR");std::string path=std::string(dir)+"/"+src->name+".q3g32";
 int fd=open(path.c_str(),O_RDONLY);if(fd<0)return false;q3g32_file_header h{};struct stat st{};
 size_t bytes=ggml_nbytes(src)/18*14;
 bool ok=fstat(fd,&st)==0&&pread(fd,&h,sizeof h,0)==sizeof h&&!memcmp(h.magic,"HTPQ3G1",7)&&h.version==1&&h.bits==3&&h.group==32&&h.tile_bytes==448&&h.k==(uint64_t)src->ne[0]&&h.n==(uint64_t)src->ne[1]&&h.bytes==bytes&&st.st_size==(off_t)(4096+bytes);
 const int64_t begin=ggml_time_us();size_t done=0,calls=0;
 if(ok)ok=posix_fadvise(fd,4096,bytes,POSIX_FADV_DONTNEED)==0;
 while(ok&&done<bytes){ssize_t n=pread(fd,(uint8_t*)dst+done,bytes-done,4096+done);++calls;if(n<0&&errno==EINTR)continue;if(n<=0){ok=false;break;}done+=n;}
 LLAMA_LOG_INFO("q3g32 read tensor=%s payload=%zu carrier=%zu begin_us=%lld end_us=%lld calls=%zu ok=%d\n",src->name,bytes,ggml_nbytes(src),(long long)begin,(long long)ggml_time_us(),calls,ok);
 if(!ok||!q3g32_register_storage(dst,bytes)){close(fd);return false;}std::lock_guard<std::mutex>lock(q3g32_mutex);q3g32_jobs[ctx].push_back({fd,bytes,src->name});return true;
}
static bool q3g32_present(void*ctx){std::lock_guard<std::mutex>lock(q3g32_mutex);return q3g32_jobs.count(ctx)!=0;}
static void q3g32_after_sync(void*ctx){
 std::vector<q3g32_pending>jobs;{std::lock_guard<std::mutex>lock(q3g32_mutex);auto it=q3g32_jobs.find(ctx);if(it==q3g32_jobs.end())return;jobs.swap(it->second);q3g32_jobs.erase(it);}
 const int64_t complete=ggml_time_us();for(auto&j:jobs)fp16_evictions().submit(j.fd,4096,j.bytes,j.name.c_str());
 LLAMA_LOG_INFO("q3g32 after-compute eviction ctx=%p tensors=%zu sync_us=%lld enqueue_end_us=%lld\n",ctx,jobs.size(),(long long)complete,(long long)ggml_time_us());
}
