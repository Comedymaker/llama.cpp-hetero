#pragma once
#include "mix-format.h"
#include <sys/uio.h>
#include <unistd.h>
#include <errno.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <chrono>

namespace mix_direct {
struct stats {uint64_t bytes=0,calls=0,iovs=0;int64_t prepare_us=0,read_us=0,validate_us=0;};
inline int64_t now_us(){return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}

// Preadv consumes consecutive FILE bytes but scatters them into noncontiguous
// final rpcmem destinations. No packed payload buffer and no payload memcpy.
inline bool read_into(int fd,uint8_t *dst,const uint8_t *map,size_t tiles,off_t payload_offset,stats &s,size_t limit=1024){
    if(!dst||!map||!tiles||payload_offset<0||limit==0){errno=EINVAL;return false;}
    const long sys_limit=sysconf(_SC_IOV_MAX);
    limit=std::min(limit,(size_t)(sys_limit>0?sys_limit:1024));
    static thread_local std::vector<struct iovec> vectors;
    vectors.clear();vectors.reserve(limit);
    size_t tile=0;off_t offset=payload_offset;
    while(tile<tiles){
        const int64_t prep=now_us();vectors.clear();
        size_t first=tile;
        while(tile<tiles){
            const bool hi=htp_mix_g32::q4_at(map,tile);
            uint8_t *p=dst+tile*576;const size_t bytes=hi?576:320;
            bool joins=!vectors.empty()&&(uint8_t*)vectors.back().iov_base+vectors.back().iov_len==p;
            if(!joins&&vectors.size()==limit)break;
            if(!hi){memset(p+320,0,256);p[574]=0x55;p[575]=0x7e;}
            if(joins)vectors.back().iov_len+=bytes;
            else vectors.push_back({p,bytes});
            ++tile;
        }
        s.iovs+=vectors.size();s.prepare_us+=now_us()-prep;
        size_t head=0;
        const int64_t read_start=now_us();
        while(head<vectors.size()){
            ++s.calls;
            const ssize_t n=preadv(fd,vectors.data()+head,(int)(vectors.size()-head),offset);
            if(n<0&&errno==EINTR)continue;
            if(n<=0){s.read_us+=now_us()-read_start;return false;}
            size_t used=(size_t)n;offset+=n;s.bytes+=used;
            while(head<vectors.size()&&used>=vectors[head].iov_len){used-=vectors[head].iov_len;++head;}
            if(used){if(head==vectors.size()){errno=EIO;return false;}vectors[head].iov_base=(uint8_t*)vectors[head].iov_base+used;vectors[head].iov_len-=used;}
        }
        s.read_us+=now_us()-read_start;
        const int64_t validate_start=now_us();
        for(size_t t=first;t<tile;++t){
            if(htp_mix_g32::q4_at(map,t)&&!htp_mix_g32::q4_scale_finite(dst+t*576)){
                s.validate_us+=now_us()-validate_start;errno=EILSEQ;return false;
            }
        }
        s.validate_us+=now_us()-validate_start;
    }
    return true;
}
} // namespace mix_direct
