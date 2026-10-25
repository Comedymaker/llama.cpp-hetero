#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
int main(int argc,char **argv){
 if(argc!=2)return 64;
 const char *names[]={"attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"};
 std::atomic<int> good{0};std::vector<std::thread> threads;
 for(int i=0;i<7;++i)threads.emplace_back([&,i]{
  std::string path=std::string(argv[1])+"/blk.8."+names[i]+".weight.q4tile";
  int fd=open(path.c_str(),O_RDONLY);if(fd<0)return;
  struct header{char magic[8];uint32_t version,type;uint64_t k,n,bytes;}h{};
  if(read(fd,&h,sizeof h)!=sizeof h || memcmp(h.magic,"HTPQ4T1\0",8)){close(fd);return;}
  std::vector<unsigned char> expected(h.bytes),actual(h.bytes);
  if(lseek(fd,4096,SEEK_SET)<0){close(fd);return;}size_t done=0;
  while(done<h.bytes){ssize_t n=read(fd,expected.data()+done,h.bytes-done);if(n<=0){close(fd);return;}done+=n;}
  for(int rep=0;rep<3;++rep){memset(actual.data(),0,actual.size());
   if(pread(fd,actual.data(),actual.size(),4096)!=ssize_t(actual.size()) || memcmp(actual.data(),expected.data(),actual.size())){close(fd);return;}
   ++good;
  }
  close(fd);
 });
 for(auto &t:threads)t.join();fprintf(stderr,"POOL_BYTE_TEST exact=%d expected=21\n",good.load());return good==21?0:1;
}
