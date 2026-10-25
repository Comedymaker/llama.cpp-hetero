// Assemble frozen existing Q2/Q3 tiles. No quantization or FP16 source read.
#include "format/compact-format.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
using namespace htp_mix_compact;
static void ck(bool x,const char*m){if(!x){fprintf(stderr,"FAIL %s errno=%d\n",m,errno);exit(1);}}
static void rx(int fd,void*p,size_t n,size_t at){for(size_t d=0;d<n;){ssize_t r=pread(fd,(char*)p+d,n-d,at+d);if(r<0&&errno==EINTR)continue;ck(r>0,"read");d+=r;}}
static void wx(int fd,const void*p,size_t n){for(size_t d=0;d<n;){ssize_t r=write(fd,(char*)p+d,n-d);if(r<0&&errno==EINTR)continue;ck(r>0,"write");d+=r;}}
struct original{char magic[8];uint32_t version,bits,group,tile;uint64_t k,n,bytes;};
static std::vector<uint8_t> load(std::string path,int bits,original&h){int fd=open(path.c_str(),O_RDONLY);ck(fd>=0,"source open");struct stat st{};ck(!fstat(fd,&st),"source stat");rx(fd,&h,sizeof h,0);ck(!memcmp(h.magic,bits==2?"HTPQ2G1":"HTPQ3G1",7)&&h.version==1&&h.bits==(unsigned)bits&&h.group==32&&h.tile==(bits==2?320:448)&&h.k%32==0&&h.n%32==0&&h.bytes==h.k*h.n/1024*h.tile&&st.st_size==off_t(4096+h.bytes),"source ABI");std::vector<uint8_t>v(h.bytes);rx(fd,v.data(),v.size(),4096);close(fd);return v;}
int main(int ac,char**av){
 ck(ac==5,"usage Q2dir Q3dir assignments.tsv FRESHoutdir");setvbuf(stdout,nullptr,_IONBF,0);
 int bit[32][32]={};FILE*f=fopen(av[3],"r");ck(f,"assignments");char line[128];ck(fgets(line,sizeof line,f),"assignment header read");line[strcspn(line,"\r\n")]=0;ck(!strcmp(line,"layer\tshard\tbits"),"assignment header");int high=0,total=0;
 while(fgets(line,sizeof line,f)){int l,s,b;ck(sscanf(line,"%d%d%d",&l,&s,&b)==3&&l>=8&&l<32&&s>=0&&s<32&&(b==2||b==3)&&!bit[l][s],"unique assignment");bit[l][s]=b;high+=b==3;++total;}fclose(f);ck(total==768&&high==573,"frozen 573Q3/195Q2");ck(!mkdir(av[4],0755),"new output directory");
 const char*suffix[]={"attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"};uint64_t all_payload=0,all_desc=0,high_tiles=0;int count=0;
 for(int l=8;l<32;l++)for(int j=0;j<7;j++){
  std::string name="blk."+std::to_string(l)+"."+suffix[j]+".weight";original a{},b{};auto q2=load(std::string(av[1])+"/"+name+".q2g32",2,a);auto q3=load(std::string(av[2])+"/"+name+".q3g32",3,b);ck(a.k==b.k&&a.n==b.n,"source dimensions");size_t nk=a.k/32,tiles=nk*(a.n/32),desc=descriptor_padded(tiles);std::vector<uint8_t>data(desc,0);size_t payload=0,nhi=0;
  for(size_t t=0;t<tiles;t++){size_t ct=t/nk,kt=t%nk;size_t shard=(j==3||j==6)?(kt*32)/(a.k/32):(ct*32)/(a.n/32);ck(shard<32,"shard geometry");bool hi=bit[l][shard]==3;uint32_t d=uint32_t(payload)|(hi?kQ4Flag:0);memcpy(data.data()+t*4,&d,4);const size_t width=hi?448:320,at=data.size();data.resize(at+width);memcpy(data.data()+at,(hi?q3.data():q2.data())+t*width,width);payload+=width;nhi+=hi;}
  file_header h{};memcpy(h.magic,"HTPCM23\0",8);h.version=1;h.bits_low=2;h.bits_high=3;h.group=32;h.k=a.k;h.n=a.n;h.tiles=tiles;h.payload_bytes=payload;h.descriptor_bytes=desc;h.data_bytes=data.size();ck(valid_header(h,h.k,h.n,4096+h.data_bytes)&&validate_data(h,data.data()),"new Q23 compact valid");
  std::string path=std::string(av[4])+"/"+name+".mixcompact";int out=open(path.c_str(),O_RDWR|O_CREAT|O_EXCL,0644);ck(out>=0,"new output");std::vector<uint8_t>page(4096,0);memcpy(page.data(),&h,sizeof h);wx(out,page.data(),page.size());wx(out,data.data(),data.size());ck(!fsync(out),"fsync");std::vector<uint8_t>back(data.size());rx(out,back.data(),back.size(),4096);ck(back==data,"byte readback");close(out);all_payload+=payload;all_desc+=desc;high_tiles+=nhi;
  printf("assembled=%d tensor=%s q3tiles=%zu q2tiles=%zu payload=%zu descriptors=%zu shared=%zu readback=EXACT\n",++count,name.c_str(),nhi,tiles-nhi,payload,desc,data.size());
 }
 ck(count==168&&all_payload==2123956224ULL&&all_desc==20447232ULL&&high_tiles==573ULL*6656,"global payload/selection");printf("COMPLETE Q23 frozen573 tensors=168 payload=%llu descriptors=%llu shared=%llu NO_REQUANTIZATION\n",(unsigned long long)all_payload,(unsigned long long)all_desc,(unsigned long long)(all_payload+all_desc));
}
