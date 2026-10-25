#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
inline int gain_tile_count(int rows,int tile){return tile>0?(std::max(0,rows)+tile-1)/tile:0;}
inline double gain_shape_cost(int rows,int leaves,double node_ms,double leaf_ms,int tile,double tile_ms){
 return std::max(0,rows)*node_ms+std::max(0,leaves)*leaf_ms+gain_tile_count(rows,tile)*tile_ms;
}
