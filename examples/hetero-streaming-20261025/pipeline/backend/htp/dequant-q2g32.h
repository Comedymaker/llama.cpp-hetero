// Custom symmetric2-bit group32. Tile256code bytes+32FP16 scales=320bytes.
// File byte[cp*32+r] holds k4cp..4cp+3 LSB-first; values=(code-2)*scale.
// HVX 2-bit extraction, pair interleave, 64-byte group shuffle, FP16 scale.
static void dequantize_tiled_weight_to_fp16_task_q2_g32(
 const tiled_dequantize_state_t*s,uint32_t first,uint32_t last){
 for(uint32_t t=first;t<last;++t){
  const uint8_t*p=s->src+(size_t)t*s->aligned_tile_size;
  HVX_Vector scales=hvx_vmem(p+256);
  HVX_Vector scale2=Q6_V_lo_W(Q6_W_vshuff_VVR(scales,scales,-2));
  __fp16*out=s->dst+(size_t)t*1024;
  for(unsigned half=0;half<2;++half){
   const HVX_Vector codes=hvx_vmem(p+128*half);
   const HVX_Vector mask=Q6_Vb_vsplat_R(3),two=Q6_Vb_vsplat_R(2);
   HVX_Vector v0=Q6_Vb_vsub_VbVb(Q6_V_vand_VV(codes,mask),two);
   HVX_Vector v1=Q6_Vb_vsub_VbVb(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(codes,2),mask),two);
   HVX_Vector v2=Q6_Vb_vsub_VbVb(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(codes,4),mask),two);
   HVX_Vector v3=Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(codes,6),two);
   HVX_VectorPair p01=Q6_W_vshuff_VVR(v1,v0,-1),p23=Q6_W_vshuff_VVR(v3,v2,-1);
   HVX_VectorPair cp01=Q6_W_vshuff_VVR(Q6_V_lo_W(p23),Q6_V_lo_W(p01),-64);
   HVX_VectorPair cp23=Q6_W_vshuff_VVR(Q6_V_hi_W(p23),Q6_V_hi_W(p01),-64);
   HVX_Vector values[4]={Q6_V_lo_W(cp01),Q6_V_hi_W(cp01),Q6_V_lo_W(cp23),Q6_V_hi_W(cp23)};
   for(unsigned part=0;part<4;++part){
    HVX_VectorPair unpacked=Q6_Wh_vunpack_Vb(values[part]);
    HVX_Vector a=Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(unpacked)),scale2));
    HVX_Vector b=Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(unpacked)),scale2));
    hvx_vmem(out+half*512+128*part)=a;hvx_vmem(out+half*512+128*part+64)=b;
   }
  }
 }
}
