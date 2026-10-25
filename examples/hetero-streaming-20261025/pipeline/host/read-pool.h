#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>
class q4_read_pool {
 std::mutex mutex;
 std::condition_variable ready,done;
 std::function<void(unsigned)> task;
 std::vector<std::thread> workers;
 uint64_t epoch=0;
 unsigned completed=0;
 bool stop=false;
public:
 q4_read_pool(unsigned participants=7,const char*purpose="codes"):participants(participants){
  try{
   for(unsigned id=1;id<participants;++id)workers.emplace_back([this,id,purpose]{
    LLAMA_LOG_INFO("q4 tensor worker ready role=%s participant=%u participants=%u tid=%ld persistent=1\n",purpose,id,this->participants,(long)syscall(SYS_gettid));
    uint64_t seen=0;
    for(;;){
     std::unique_lock<std::mutex> lock(mutex);
     ready.wait(lock,[&]{return stop||epoch!=seen;});if(stop)return;
     seen=epoch;auto fn=task;lock.unlock();fn(id);lock.lock();
     ++completed;done.notify_one();
    }
   });
  }catch(...){shutdown();throw;}
 }
 void shutdown(){
  {std::lock_guard<std::mutex> lock(mutex);stop=true;}ready.notify_all();
  for(auto &worker:workers)if(worker.joinable())worker.join();
 }
 ~q4_read_pool(){shutdown();}
 void run(const std::function<void(unsigned)> &fn){
  {std::lock_guard<std::mutex> lock(mutex);task=fn;completed=0;++epoch;}
  ready.notify_all();fn(0);
  std::unique_lock<std::mutex> lock(mutex);done.wait(lock,[&]{return completed==participants-1;});task={};
 }
private:
 unsigned participants;
};
