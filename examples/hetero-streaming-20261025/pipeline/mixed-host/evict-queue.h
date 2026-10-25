#include <condition_variable>
#include <deque>
#include <thread>
#include <mutex>
class fp16_evict_queue {
 struct job {int fd;size_t base,bytes;std::string name;};
 std::mutex mutex;
 std::condition_variable ready,space;
 std::deque<job> jobs;
 bool stopping=false;
 size_t outstanding=0,count=0,peak_bytes=0,peak_jobs=0;
 uint64_t completed=0,failures=0,wait_us=0;
 std::thread worker;
 void work(){
  for(;;){
   std::unique_lock<std::mutex> lock(mutex);
   ready.wait(lock,[&]{return stopping||!jobs.empty();});
   if(jobs.empty()&&stopping)return;
   job j=std::move(jobs.front());jobs.pop_front();lock.unlock();
   const int64_t begin=ggml_time_us();
   const int rc=posix_fadvise(j.fd,j.base,j.bytes,POSIX_FADV_DONTNEED);
   const int64_t end=ggml_time_us();close(j.fd);
   LLAMA_LOG_INFO("q4 async_evict tensor=%s bytes=%zu begin_us=%lld end_us=%lld rc=%d\n",j.name.c_str(),j.bytes,(long long)begin,(long long)end,rc);
   lock.lock();outstanding-=j.bytes;--count;++completed;if(rc)++failures;lock.unlock();space.notify_all();
  }
 }
public:
 fp16_evict_queue():worker([this]{work();}){}
 void submit(int fd,size_t base,size_t bytes,const char *name){
  const int64_t begin=ggml_time_us();
  std::unique_lock<std::mutex> lock(mutex);
  space.wait(lock,[&]{return count<9&&outstanding+bytes<=512ULL*1024*1024;});
  wait_us+=ggml_time_us()-begin;
  jobs.push_back({fd,base,bytes,name});outstanding+=bytes;++count;
  peak_bytes=std::max(peak_bytes,outstanding);peak_jobs=std::max(peak_jobs,count);
  lock.unlock();ready.notify_one();
 }
 ~fp16_evict_queue(){
  {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
  ready.notify_one();worker.join();
  LLAMA_LOG_INFO("q4 async_evict summary completed=%llu failures=%llu producer_wait_us=%llu peak_pending_bytes=%zu peak_pending_jobs=%zu cap_bytes=536870912 cap_jobs=9\n",(unsigned long long)completed,(unsigned long long)failures,(unsigned long long)wait_us,peak_bytes,peak_jobs);
 }
};
static fp16_evict_queue & fp16_evictions(){static fp16_evict_queue queue;return queue;}
