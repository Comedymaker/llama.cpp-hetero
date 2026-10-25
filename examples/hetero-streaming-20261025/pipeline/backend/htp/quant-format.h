#ifndef UNIFIED_QUANT_FORMAT_H
#define UNIFIED_QUANT_FORMAT_H
// Stable storage metadata. Per-format decode code remains specialized.
enum unified_storage { UQ_ROW_BLOCKS, UQ_REPACKED_TILES };
struct unified_quant_format { unsigned type,bits,group,block_bytes;enum unified_storage storage; };
static inline const struct unified_quant_format *unified_quant_format(unsigned type){
 static const struct unified_quant_format formats[]={
  {HTP_TYPE_Q2_K,2,256,84,UQ_ROW_BLOCKS},
  {HTP_TYPE_Q3_G32,3,32,14,UQ_REPACKED_TILES},
  {HTP_TYPE_MIX_G32,4,32,18,UQ_REPACKED_TILES},
  {HTP_TYPE_Q4_0,4,32,18,UQ_REPACKED_TILES},
  {HTP_TYPE_Q4_1,4,32,20,UQ_REPACKED_TILES},
  {HTP_TYPE_Q8_0,8,32,34,UQ_REPACKED_TILES},
  {HTP_TYPE_IQ4_NL,4,32,18,UQ_REPACKED_TILES},
  {HTP_TYPE_MXFP4,4,32,17,UQ_REPACKED_TILES}
 };
 for(unsigned i=0;i<sizeof(formats)/sizeof(formats[0]);++i)if(formats[i].type==type)return &formats[i];
 return NULL;
}
static inline int unified_is_tile_storage(unsigned type){
 const struct unified_quant_format*f=unified_quant_format(type);return f&&f->storage==UQ_REPACKED_TILES;
}
#endif
