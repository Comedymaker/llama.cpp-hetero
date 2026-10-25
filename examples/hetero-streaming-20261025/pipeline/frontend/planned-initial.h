#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cstring>

// Optional explicit planner profile. Never silently accept a malformed file.
struct planned_initial_profile {
    bool enabled=false;
    double verify[65]={},cpu[17]={};
    planned_initial_profile(){
        const char *path=std::getenv("LLAMA_TREE_INITIAL_PROFILE");
        if(!path)return;
        FILE *f=std::fopen(path,"r");if(!f)fail("cannot open profile");
        char line[256];unsigned nv=0,nc=0;
        while(std::fgets(line,sizeof line,f)){
            if(line[0]=='#'||line[0]=='\n'||line[0]=='\r')continue;
            char kind=0,extra=0;int index=0;double ms=0;
            if(std::sscanf(line," %c %d %lf %c",&kind,&index,&ms,&extra)!=3||!std::isfinite(ms)||ms<=0)fail("invalid profile row");
            if(kind=='V'&&index>=1&&index<=64&&!verify[index]){verify[index]=ms;++nv;}
            else if(kind=='C'&&index>=1&&index<=16&&!cpu[index]){cpu[index]=ms;++nc;}
            else fail("duplicate/unknown profile row");
        }
        std::fclose(f);
        if(nv!=64||nc!=16)fail("profile requires V1..64 and C1..16");
        for(int i=2;i<=64;++i)if(verify[i]<verify[i-1])fail("verification curve must be monotone");
        enabled=true;
        std::fprintf(stderr,"tree initial planner profile=%s verify-points=64 cpu-points=16 leaves=not-independently-fitted online-update=unchanged\n",path);
    }
    [[noreturn]] static void fail(const char *message){std::fprintf(stderr,"tree initial profile ERROR: %s\n",message);std::abort();}
    double verification(int rows)const{return verify[std::max(1,std::min(64,rows))];}
};
static const planned_initial_profile & planned_initial(){static planned_initial_profile p;return p;}
