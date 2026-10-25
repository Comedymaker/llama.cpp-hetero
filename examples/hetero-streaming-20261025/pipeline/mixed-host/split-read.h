#include <sys/uio.h>
#include <map>
#include <mutex>
#include <vector>
struct q4_split_job {int codes=-1,scales=-1;uint8_t*dst=nullptr;size_t tiles=0;std::string name;bool scales_ready=false;};
static std::mutex q4_split_mutex;
static std::map<void*,std::vector<q4_split_job>> q4_split_jobs;
static bool q4_scatter_read(int fd,uint8_t*dst,size_t tiles,bool scale,const char*name){
 size_t width=scale?64:512;const int64_t begin=ggml_time_us();size_t calls=0;
 uint8_t *target=dst+(scale?tiles*512:0);size_t total=tiles*width,done=0;
 while(done<total){ssize_t got=pread(fd,target+done,total-done,4096+done);++calls;
  if(got<0&&errno==EINTR)continue;if(got<=0)return false;done+=got;
 }
 const int64_t end=ggml_time_us();
 LLAMA_LOG_INFO("split read part=%s tensor=%s bytes=%zu calls=%zu begin_us=%lld end_us=%lld time_us=%lld staging=0 contiguous=1\n",scale?"scale":"codes",name,tiles*width,calls,(long long)begin,(long long)end,(long long)(end-begin));
 return true;
}
static bool q4_split_read(void*ctx,const ggml_tensor*src,void*dst){
 const char*dir=std::getenv("LLAMA_Q4_SPLIT_DIR");std::string p=std::string(dir)+"/"+src->name+".q4tile";
 q4_split_job j;j.codes=open((p+".codes").c_str(),O_RDONLY);j.scales=open((p+".scales").c_str(),O_RDONLY);j.dst=(uint8_t*)dst;j.name=src->name;j.tiles=ggml_nbytes(src)/576;
 bool ok=j.codes>=0&&j.scales>=0&&ggml_nbytes(src)%576==0;
 struct header {char magic[8];uint32_t version,type;uint64_t k,n,bytes;};
 for(int fd:{j.codes,j.scales}){header h{};struct stat st{};bool sc=fd==j.scales;
  ok=ok&&fstat(fd,&st)==0&&pread(fd,&h,sizeof h,0)==sizeof h&&!memcmp(h.magic,"HTPQ4T1",7)&&h.version==1&&h.type==GGML_TYPE_Q4_0&&h.bytes==ggml_nbytes(src)&&h.k==(uint64_t)src->ne[0]&&h.n==(uint64_t)src->ne[1]&&st.st_size==(off_t)(4096+j.tiles*(sc?64:512));
 }
 // Only codes are invalidated/read during prefetch. Scale invalidation is deferred too.
 if(ok)ok=posix_fadvise(j.codes,4096,j.tiles*512,POSIX_FADV_DONTNEED)==0&&q4_scatter_read(j.codes,j.dst,j.tiles,false,src->name);
 int layer=-1;sscanf(src->name,"blk.%d.",&layer);
 if(ok&&layer==8){ok=posix_fadvise(j.scales,4096,j.tiles*64,POSIX_FADV_DONTNEED)==0&&q4_scatter_read(j.scales,j.dst,j.tiles,true,src->name);j.scales_ready=ok;}
 if(!ok){if(j.codes>=0)close(j.codes);if(j.scales>=0)close(j.scales);return false;}
 {std::lock_guard<std::mutex>l(q4_split_mutex);q4_split_jobs[ctx].push_back(std::move(j));}
 return true;
}
static bool q4_split_finish_scales(void*ctx,bool &present){
 std::vector<q4_split_job> jobs;
 {std::lock_guard<std::mutex>l(q4_split_mutex);auto it=q4_split_jobs.find(ctx);present=it!=q4_split_jobs.end();if(present)jobs=it->second;}
 if(!present)return true;
 const int64_t begin=ggml_time_us();
 std::atomic<size_t> next{0};std::atomic<bool> failed{false};
 static thread_local q4_read_pool scales_pool(4,"scale");
 scales_pool.run([&](unsigned){
  while(!failed.load()){
   size_t i=next.fetch_add(1);if(i>=jobs.size())break;auto&j=jobs[i];
   if(!j.scales_ready&&(posix_fadvise(j.scales,4096,j.tiles*64,POSIX_FADV_DONTNEED)!=0||!q4_scatter_read(j.scales,j.dst,j.tiles,true,j.name.c_str())))failed.store(true);
  }
 });
 if(failed.load())return false;
 LLAMA_LOG_INFO("split scales ready ctx=%p tensors=%zu begin_us=%lld end_us=%lld\n",ctx,jobs.size(),(long long)begin,(long long)ggml_time_us());return true;
}
static void q4_split_after_sync(void*ctx){
 std::vector<q4_split_job>jobs;
 {std::lock_guard<std::mutex>l(q4_split_mutex);auto it=q4_split_jobs.find(ctx);if(it==q4_split_jobs.end())return;jobs.swap(it->second);q4_split_jobs.erase(it);}
 const int64_t complete=ggml_time_us();
 for(auto&j:jobs){fp16_evictions().submit(j.codes,4096,j.tiles*512,j.name.c_str());fp16_evictions().submit(j.scales,4096,j.tiles*64,j.name.c_str());}
 LLAMA_LOG_INFO("split after-compute eviction ctx=%p tensors=%zu compute_complete_us=%lld enqueue_end_us=%lld\n",ctx,jobs.size(),(long long)complete,(long long)ggml_time_us());
}
