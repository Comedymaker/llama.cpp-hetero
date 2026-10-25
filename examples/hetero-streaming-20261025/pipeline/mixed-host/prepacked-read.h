#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <sys/stat.h>
#include "evict-queue.h"
#include "split-read.h"
#include "q3g32-read.h"
#include "../format/mix-read.h"
#include "../format/compact-read.h"
static bool q4_prepacked_read(const ggml_tensor *src,void *dst,void*ctx){
 if(std::getenv("LLAMA_MIX_COMPACT_DIR"))return mixcompact_read(ctx,src,dst);
 if(std::getenv("LLAMA_MIX_G32_DIR"))return mixg32_read(ctx,src,dst);
 if(std::getenv("LLAMA_Q3_GROUP32_DIR"))return q3g32_read(ctx,src,dst);
 if(std::getenv("LLAMA_Q4_SPLIT_DIR"))return q4_split_read(ctx,src,dst);
 struct header { char magic[8];uint32_t version,type;uint64_t k,n,bytes; } h{};
 const char *dir=std::getenv("LLAMA_Q4_PREPACK_DIR");
 std::string path=std::string(dir)+"/"+src->name+".q4tile";
 int fd=open(path.c_str(),O_RDONLY);if(fd<0)return false;
 struct stat st{};bool valid=fstat(fd,&st)==0&&pread(fd,&h,sizeof h,0)==sizeof h;
 const size_t bytes=ggml_nbytes(src),padded=(bytes+4095)&~size_t(4095);
 valid=valid&&!memcmp(h.magic,"HTPQ4T1\0",8)&&h.version==1&&h.type==GGML_TYPE_Q4_0&&h.k==(uint64_t)src->ne[0]&&h.n==(uint64_t)src->ne[1]&&h.bytes==bytes&&st.st_size==(off_t)(4096+padded);
 if(!valid){close(fd);return false;}
 const int64_t before_begin=ggml_time_us();int before_rc=posix_fadvise(fd,4096,padded,POSIX_FADV_DONTNEED);
 const int64_t begin=ggml_time_us();size_t loaded=0;
 while(loaded<bytes){ssize_t got=pread(fd,(uint8_t*)dst+loaded,bytes-loaded,4096+loaded);if(got<0&&errno==EINTR)continue;if(got<=0){int saved=errno;close(fd);LLAMA_LOG_ERROR("q4 pread HTP failed tensor=%s loaded=%zu errno=%d\n",src->name,loaded,saved);return false;}loaded+=(size_t)got;}
 const int64_t read_end=ggml_time_us();fp16_evictions().submit(fd,4096,padded,src->name);const int64_t end=ggml_time_us();
 LLAMA_LOG_INFO("q4 prepacked tensor=%s bytes=%zu before_us=%lld read_us=%lld enqueue_us=%lld copy_us=0 before_rc=%d before_begin_us=%lld begin_us=%lld read_end_us=%lld end_us=%lld ok=%d route=buffered-pread-to-HTP-async-evict staging=0\n",src->name,bytes,(long long)(begin-before_begin),(long long)(read_end-begin),(long long)(end-read_end),before_rc,(long long)before_begin,(long long)begin,(long long)read_end,(long long)end,before_rc==0);
 return before_rc==0;
}
