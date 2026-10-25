#include "compact-format.h"
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>
int main(int argc,char**argv){
 if(argc!=2)return 64;
 const char*names[]={"attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"};
 int good=0;
 for(int layer=8;layer<32;++layer)for(auto name:names){
  std::string path=std::string(argv[1])+"/blk."+std::to_string(layer)+"."+name+".weight.mixcompact";
  int fd=open(path.c_str(),O_RDONLY); if(fd<0)return 1;
  htp_mix_compact::file_header h{};
  if(read(fd,&h,sizeof(h))!=sizeof(h)||h.data_bytes%4096)return 2;
  std::vector<unsigned char>ref(h.data_bytes),got(h.data_bytes+256,0xa5);
  if(lseek(fd,4096,SEEK_SET)<0)return 3;
  size_t done=0;while(done<h.data_bytes){ssize_t n=read(fd,ref.data()+done,h.data_bytes-done);if(n<=0)return 4;done+=n;}
  if(pread(fd,got.data()+128,h.data_bytes,4096)!=ssize_t(h.data_bytes)||memcmp(ref.data(),got.data()+128,h.data_bytes))return 5;
  for(int i=0;i<128;++i)if(got[i]!=0xa5||got[128+h.data_bytes+i]!=0xa5)return 6;
  if(!htp_mix_compact::validate_data(h,got.data()+128))return 7;
  close(fd);++good;
 }
 fprintf(stderr,"STAGED_COMPACT_BYTE_TEST exact=%d expected=168 guards=128\n",good);return good==168?0:8;
}
