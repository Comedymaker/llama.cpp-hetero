// Standard GGML Q2_K raw rows -> the same FP16 HMX tiles as all other formats.
// Scalar unpack/reference-order prep, HVX FP32 arithmetic, single FP16 rounding.
// Initial correctness implementation, not a tuned LUT kernel.
static void dequantize_tiled_weight_to_fp16_task_q2_k(
 const tiled_dequantize_state_t*s,uint32_t begin,uint32_t end){
 for(uint32_t t=begin;t<end;++t){
  uint32_t ct=t/s->n_k_tiles,kt=t%s->n_k_tiles;
  __fp16*out=s->dst+(size_t)t*1024;
  for(unsigned r=0;r<32;++r){
   const uint8_t*b=s->src+(ct*32+r)*s->row_stride+(kt/8)*84;
   __fp16 hd,hm;memcpy(&hd,b+80,2);memcpy(&hm,b+82,2);
   float d=hd,dm=hm;
   float q[32] __attribute__((aligned(128)));
   float ds[32] __attribute__((aligned(128)));
   float ms[32] __attribute__((aligned(128)));
   float values[32] __attribute__((aligned(128)));
   unsigned slice=kt%8;
   for(unsigned x=0;x<32;++x){
    uint8_t sc=b[2*slice+x/16];unsigned code=(b[16+32*(slice/4)+x]>>(2*(slice%4)))&3;
    q[x]=(float)code;ds[x]=d*(sc&15);ms[x]=dm*(sc>>4);
   }
   // Preserve CPU reference order: (d*scale)*code - dmin*min, not fused FMA.
   HVX_Vector v;
#if __HVX_ARCH__ >= 79
   v=Q6_Vsf_vsub_VsfVsf(Q6_Vsf_vmpy_VsfVsf(hvx_vmem(ds),hvx_vmem(q)),hvx_vmem(ms));
#else
   HVX_Vector product=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(hvx_vmem(ds),hvx_vmem(q)));
   v=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(product,hvx_vmem(ms)));
#endif
   hvx_vmem(values)=v;
   for(unsigned x=0;x<32;++x)out[(x/2)*64+r*2+x%2]=(__fp16)values[x];
  }
 }
}
