// Experiment-only interposer. Original frontend, model libraries and DSP stay intact.
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <deque>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <sys/syscall.h>

using pread_fn = ssize_t (*)(int, void *, size_t, off_t);
static pread_fn original() {
    static auto p = reinterpret_cast<pread_fn>(dlsym(RTLD_NEXT, "pread"));
    if (!p) _exit(120);
    return p;
}
static int64_t now_us() { return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::atomic<uint64_t> calls{0}, failures{0}, logical{0}, physical{0};
static std::atomic<uint64_t> file_us{0}, copy_us{0}, ready_us{0}, overlap_est_us{0};
static constexpr size_t alignment = 4096, chunk = 8*1024*1024;
struct buffers {
    void *p[2]{nullptr,nullptr};
    ~buffers(){free(p[0]);free(p[1]);}
    bool init() {
        if (p[0] && p[1]) return true;
        if (posix_memalign(&p[0],alignment,chunk) || posix_memalign(&p[1],alignment,chunk)) {free(p[0]);free(p[1]);p[0]=p[1]=nullptr;return false;}
        return true;
    }
};
struct job {
    std::mutex m;std::condition_variable result;bool complete=false;
    int fd=-1,error=0,worker=-1;void *dst=nullptr;const void *src=nullptr;
    size_t size=0,chunk_id=0;off_t offset=0;uint64_t call_id=0;
    ssize_t got=-1;int64_t duration=0,begin=0,end=0,submitted=0;
    bool wait(int64_t &t,int &e) {
        std::unique_lock<std::mutex> l(m);result.wait(l,[this]{return complete;});
        t=duration;e=error;return got==ssize_t(size);
    }
};
struct pool {
    std::mutex m;std::condition_variable command;bool stop=false;
    std::deque<std::shared_ptr<job>> queue;
    std::vector<std::thread> workers;bool reads;
    pool(int count,bool r):reads(r) {
        try {for(int i=0;i<count;++i)workers.emplace_back([this,i]{run(i);});}
        catch(...){shutdown();throw;}
    }
    void shutdown(){ {std::lock_guard<std::mutex> l(m);stop=true;}command.notify_all();for(auto &t:workers)if(t.joinable())t.join(); }
    ~pool(){shutdown();}
    void submit(const std::shared_ptr<job> &j) {
        j->submitted=now_us();{std::lock_guard<std::mutex> l(m);queue.push_back(j);}command.notify_one();
    }
    void run(int worker) {
        fprintf(stderr,"STAGED_WORKER phase=%s worker=%d tid=%ld persistent=1\n",reads?"read":"copy",worker,(long)syscall(SYS_gettid));
        for(;;) {
            std::shared_ptr<job> j;
            {std::unique_lock<std::mutex> l(m);command.wait(l,[this]{return stop||!queue.empty();});
             if(stop&&queue.empty())return;j=queue.front();queue.pop_front();}
            const int64_t start=now_us();ssize_t g;
            if(reads){do {g=original()(j->fd,j->dst,j->size,j->offset);}while(g<0&&errno==EINTR);}
            else {memcpy(j->dst,j->src,j->size);g=j->size;}
            int e=g<0?errno:0;int64_t finish=now_us();
            {std::lock_guard<std::mutex> l(j->m);j->worker=worker;j->begin=start;j->end=finish;j->got=g;j->error=e;j->duration=finish-start;j->complete=true;}
            j->result.notify_one();
        }
    }
};
static int count_env(const char *name){const char *v=getenv(name);if(!v)_exit(122);int n=atoi(v);if(n<1||n>7)_exit(122);return n;}
struct pools {
    pool read,copy;
    pools():read(count_env("STAGED_READ_WORKERS"),true),copy(count_env("STAGED_COPY_WORKERS"),false) {
        int r=count_env("STAGED_READ_WORKERS"),c=count_env("STAGED_COPY_WORKERS");if(!((r==7&&c==1)||(r==4&&c==4)))_exit(124);
        fprintf(stderr,"STAGED_POOLS read=%d copy=%d coordinator_limit=7 staging_limit=117440512\n",count_env("STAGED_READ_WORKERS"),count_env("STAGED_COPY_WORKERS"));
    }
};
static pools &global_pools(){static pools p;return p;}
static std::shared_ptr<job> enqueue(pool &p,int fd,void *dst,const void *src,size_t n,off_t off,uint64_t id,size_t part){
    auto j=std::make_shared<job>();j->fd=fd;j->dst=dst;j->src=src;j->size=n;j->offset=off;j->call_id=id;j->chunk_id=part;p.submit(j);return j;
}
static bool tensor_path(int fd, std::string &out) {
    char proc[64],path[4096];snprintf(proc,sizeof(proc),"/proc/self/fd/%d",fd);
    ssize_t n=readlink(proc,path,sizeof(path)-1);if(n<0)return false;path[n]=0;
    const char *base=getenv("LLAMA_Q4_PREPACK_DIR");if(!base)return false;
    size_t b=strlen(base);out.assign(path,n);
    return out.size()>b+8 && out.compare(0,b,base)==0 && out[b]=='/' && out.compare(out.size()-7,7,".q4tile")==0;
}
static ssize_t transfer(int fd,void *dst,size_t count,off_t offset) {
    const char *mode=getenv("STAGED_Q4_MODE");
    if(!mode || (strcmp(mode,"serial")&&strcmp(mode,"pipe")) || offset!=4096 || count<4096)
        return original()(fd,dst,count,offset);
    const int64_t whole_begin=now_us();
    std::string path;if(!tensor_path(fd,path))return original()(fd,dst,count,offset);
    struct header{char magic[8];uint32_t version,type;uint64_t k,n,bytes;} h{};
    struct stat st{};
    size_t padded=(count+4095)&~size_t(4095);
    bool valid=original()(fd,&h,sizeof(h),0)==sizeof(h) && !memcmp(h.magic,"HTPQ4T1\0",8) && h.version==1 &&
        (h.type==2||h.type==8) && h.bytes==count && fstat(fd,&st)==0 && st.st_size==off_t(4096+padded);
    const uint64_t id=++calls;
    if(!valid){++failures;errno=EINVAL;return -1;}
    static thread_local buffers memory;
    if(!memory.init()){++failures;errno=ENOMEM;return -1;}
    int direct=open(path.c_str(),O_RDONLY|O_DIRECT|O_CLOEXEC);
    if(direct<0){++failures;return -1;}
    const bool pipe=!strcmp(mode,"pipe");if(!pipe)_exit(123);
    auto &p=global_pools();
    const int64_t union_begin=now_us();
    int64_t tr=0,tc=0;int saved=0;bool ok=true;
    size_t chunks=(padded+chunk-1)/chunk;
    std::vector<std::shared_ptr<job>> records;
    if(pipe) {
        auto current=enqueue(p.read,direct,memory.p[0],nullptr,std::min(chunk,padded),4096,id,0);
        records.push_back(current);
        int64_t t=0;ok=current->wait(t,saved);tr+=t;
        for(size_t i=0;ok&&i<chunks;++i) {
            bool next=i+1<chunks;
            std::shared_ptr<job> next_job;
            if(next)next_job=enqueue(p.read,direct,memory.p[(i+1)&1],nullptr,std::min(chunk,padded-(i+1)*chunk),4096+(i+1)*chunk,id,i+1);
            if(next)records.push_back(next_job);
            size_t bytes=std::min(chunk,count-i*chunk);
            auto copying=enqueue(p.copy,-1,static_cast<char*>(dst)+i*chunk,memory.p[i&1],bytes,0,id,i);
            records.push_back(copying);
            ok=copying->wait(t,saved);tc+=t;
            if(next){int error=0;bool read_ok=next_job->wait(t,error);tr+=t;if(!read_ok){ok=false;saved=error;}}
        }
    }else{
        for(size_t i=0;ok&&i<chunks;++i) {
            size_t bytes=std::min(chunk,padded-i*chunk);int64_t start=now_us();ssize_t got;
            do{got=original()(direct,memory.p[i&1],bytes,4096+i*chunk);}while(got<0&&errno==EINTR);
            int64_t finish=now_us();tr+=finish-start;
            fprintf(stderr,"STAGED_INTERVAL phase=read call=%llu chunk=%zu begin_us=%lld end_us=%lld bytes=%zu ok=%d\n",(unsigned long long)id,i,(long long)start,(long long)finish,bytes,got==ssize_t(bytes));
            if(got!=ssize_t(bytes)){ok=false;saved=got<0?errno:EIO;break;}
            size_t copied=std::min(bytes,count-i*chunk);
            start=now_us();memcpy(static_cast<char*>(dst)+i*chunk,memory.p[i&1],copied);finish=now_us();tc+=finish-start;
            fprintf(stderr,"STAGED_INTERVAL phase=copy call=%llu chunk=%zu begin_us=%lld end_us=%lld bytes=%zu ok=1\n",(unsigned long long)id,i,(long long)start,(long long)finish,copied);
        }
    }
    int64_t union_end=now_us();close(direct);int64_t whole_end=now_us();
    // Emit only after all jobs complete: stderr locks cannot hold a read/copy worker.
    for(const auto &j:records)fprintf(stderr,"STAGED_INTERVAL phase=%s call=%llu chunk=%zu begin_us=%lld end_us=%lld bytes=%zu ok=%d worker=%d submitted_us=%lld\n",j->src?"copy":"read",(unsigned long long)j->call_id,j->chunk_id,(long long)j->begin,(long long)j->end,j->size,j->got==ssize_t(j->size),j->worker,(long long)j->submitted);
    const int64_t overlap=std::max<int64_t>(0,tr+tc-(union_end-union_begin));
    file_us+=tr;copy_us+=tc;ready_us+=whole_end-whole_begin;overlap_est_us+=overlap;
    if(ok){logical+=count;physical+=padded;}else ++failures;
    fprintf(stderr,"STAGED_Q4 call=%llu tensor=%s mode=%s bytes=%zu padded=%zu chunks=%zu file_us=%lld copy_us=%lld union_us=%lld ready_us=%lld overlap_est_us=%lld begin_us=%lld end_us=%lld staging_per_caller=16777216 persistent_helper=%d ok=%d\n",
       (unsigned long long)id,path.substr(path.find_last_of('/')+1).c_str(),mode,count,padded,chunks,(long long)tr,(long long)tc,(long long)(union_end-union_begin),
       (long long)(whole_end-whole_begin),(long long)overlap,(long long)whole_begin,(long long)whole_end,pipe,ok);
    if(!ok){errno=saved?saved:EIO;return -1;}return count;
}
extern "C" ssize_t pread(int fd,void *dst,size_t n,off_t o){return transfer(fd,dst,n,o);}
extern "C" ssize_t __pread_chk(int fd,void *dst,size_t n,off_t o,size_t capacity){
    if(n>capacity){errno=EOVERFLOW;return -1;}return transfer(fd,dst,n,o);
}
__attribute__((destructor)) static void report(){
 fprintf(stderr,"STAGED_Q4_SUMMARY calls=%llu failures=%llu logical=%llu padded=%llu file_us=%llu copy_us=%llu ready_us=%llu overlap_est_us=%llu\n",
 (unsigned long long)calls.load(),(unsigned long long)failures.load(),(unsigned long long)logical.load(),(unsigned long long)physical.load(),
 (unsigned long long)file_us.load(),(unsigned long long)copy_us.load(),(unsigned long long)ready_us.load(),(unsigned long long)overlap_est_us.load());
}
