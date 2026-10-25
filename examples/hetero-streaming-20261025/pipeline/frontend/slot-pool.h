#include "ggml-backend.h"
#include <dlfcn.h>
#include <mutex>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdio>

// Reuse public buffer ownership, without changing CPU/HTP/DSP libraries.
// Idle buffers remain charged to the original allocation budget.
namespace slot_pool {
struct entry { ggml_backend_buffer_t buf; ggml_backend_buffer_type_t type; size_t bytes; int slot; bool live; };
struct state { std::mutex mutex; std::vector<entry> entries; uint64_t hits=0,misses=0,returns=0; size_t retained=0,peak=0; };
inline state & get() { static state * s=new state;return *s; }
thread_local int active_slot=-1;
thread_local bool bypass=false;
template<class T> T next(const char * name) { auto p=reinterpret_cast<T>(dlsym(RTLD_NEXT,name));if(!p){std::fprintf(stderr,"slot pool missing real symbol=%s\n",name);std::abort();}return p; }
using alloc_fn=ggml_backend_buffer_t(*)(ggml_backend_buffer_type_t,size_t);
using free_fn=void(*)(ggml_backend_buffer_t);
inline free_fn real_free() { static auto p=next<free_fn>("ggml_backend_buffer_free");return p; }
inline bool enabled() { const char * p=std::getenv("LLAMA_STI_SLOT_REUSE");return p&&std::strcmp(p,"1")==0; }
inline void drain() {
 auto & s=get();std::lock_guard<std::mutex> lock(s.mutex);
 for(size_t i=0;i<s.entries.size();) {
  if(s.entries[i].live){++i;continue;}
  auto e=s.entries[i];s.entries.erase(s.entries.begin()+i);s.retained-=e.bytes;
  bypass=true;real_free()(e.buf);bypass=false;
 }
}
struct scope { int old;scope(int slot):old(active_slot){active_slot=slot;}~scope(){active_slot=old;} };
struct reporter { ~reporter(){auto &s=get();std::fprintf(stderr,"slot pool summary hits=%llu misses=%llu returns=%llu retained_bytes=%zu peak_bytes=%zu entries=%zu\n",(unsigned long long)s.hits,(unsigned long long)s.misses,(unsigned long long)s.returns,s.retained,s.peak,s.entries.size());} };
static reporter report;
}

extern "C" ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t type,size_t bytes) {
 static auto real=slot_pool::next<slot_pool::alloc_fn>("ggml_backend_buft_alloc_buffer");
 if(slot_pool::bypass || !slot_pool::enabled() || slot_pool::active_slot<0 || !bytes || std::strncmp(ggml_backend_buft_name(type),"HTP",3)!=0) return real(type,bytes);
 auto &s=slot_pool::get();std::lock_guard<std::mutex> lock(s.mutex);
 size_t best=s.entries.size();
 for(size_t i=0;i<s.entries.size();++i){auto &e=s.entries[i];if(!e.live&&e.slot==slot_pool::active_slot&&e.type==type&&e.bytes>=bytes&&(best==s.entries.size()||e.bytes<s.entries[best].bytes))best=i;}
 if(best<s.entries.size()) {
  auto &e=s.entries[best];e.live=true;++s.hits;
  std::fprintf(stderr,"slot pool event=reuse slot=%d base=%p capacity=%zu request=%zu\n",e.slot,ggml_backend_buffer_get_base(e.buf),e.bytes,bytes);
  return e.buf;
 }
 // Discard undersized idle entries instead of caching every shape forever.
 for(size_t i=0;i<s.entries.size();) {
  auto e=s.entries[i];if(e.live||e.slot!=slot_pool::active_slot||e.type!=type){++i;continue;}
  s.entries.erase(s.entries.begin()+i);s.retained-=e.bytes;
  slot_pool::bypass=true;slot_pool::real_free()(e.buf);slot_pool::bypass=false;
 }
 slot_pool::bypass=true;auto buf=real(type,bytes);slot_pool::bypass=false;
 if(buf){size_t capacity=ggml_backend_buffer_get_size(buf);s.entries.push_back({buf,type,capacity,slot_pool::active_slot,true});s.retained+=capacity;s.peak=std::max(s.peak,s.retained);++s.misses;
  std::fprintf(stderr,"slot pool event=new slot=%d base=%p capacity=%zu\n",slot_pool::active_slot,ggml_backend_buffer_get_base(buf),capacity);}
 return buf;
}

extern "C" void ggml_backend_buffer_free(ggml_backend_buffer_t buf) {
 if(!buf)return;
 if(!slot_pool::bypass) {
  auto &s=slot_pool::get();std::lock_guard<std::mutex> lock(s.mutex);
  for(auto &e:s.entries)if(e.buf==buf){if(!e.live){std::fprintf(stderr,"slot pool double return\n");std::abort();}e.live=false;++s.returns;
   std::fprintf(stderr,"slot pool event=return slot=%d base=%p capacity=%zu\n",e.slot,ggml_backend_buffer_get_base(buf),e.bytes);return;}
 }
 slot_pool::real_free()(buf);
}

extern "C" void ggml_backend_free(ggml_backend_t backend) {
 using fn=void(*)(ggml_backend_t);static auto real=slot_pool::next<fn>("ggml_backend_free");
 // Free idle mappings before backend/session teardown, never active owners.
 if(!slot_pool::bypass)slot_pool::drain();
 real(backend);
}
