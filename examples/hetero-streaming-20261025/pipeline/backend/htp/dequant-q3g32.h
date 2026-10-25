// Custom symmetric Q3 group32 tile:
//   [0, 256)   low two bits, byte[cp * 32 + r] = c=4*cp..4*cp+3
//   [256, 384) high bit, byte[(c / 8) * 32 + r], bit (c & 7)
//   [384, 448) 32 fp16 scales.  The VTCM stride is 512 bytes.
// The output is the standard 32x32 HMX fp16 tile and q dequantizes as
// (q - 4) * scale.
static void dequantize_tiled_weight_to_fp16_task_q3_g32(
 const tiled_dequantize_state_t *s, uint32_t first, uint32_t last) {
 // For four consecutive code planes, duplicate each 32-byte high-bit group
 // for its two code planes.  The shift is 0/4/0/4 because planes 0 and 1
 // occupy low/high nibbles of one high-bit byte, respectively.
 static const uint8_t __attribute__((aligned(128))) high_shift_base[128] = {
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
 };
 const HVX_Vector mask2 = Q6_Vb_vsplat_R(3);
 const HVX_Vector mask1 = Q6_Vb_vsplat_R(1);
 const HVX_Vector four  = Q6_Vb_vsplat_R(4);
 const HVX_Vector shifts = hvx_vmem(high_shift_base);

 for (uint32_t t = first; t < last; ++t) {
  const uint8_t *p = s->src + (size_t)t * s->aligned_tile_size;
  const HVX_Vector scales = hvx_vmem(p + 384);
  const HVX_Vector scale2 = Q6_V_lo_W(Q6_W_vshuff_VVR(scales, scales, -2));
  __fp16 *out = s->dst + (size_t)t * 1024;
  const HVX_Vector high_all = hvx_vmem(p + 256);
  const HVX_VectorPair high_pair = Q6_W_vshuff_VVR(high_all, high_all, -32);

  for (unsigned half = 0; half < 2; ++half) {
   const HVX_Vector low = hvx_vmem(p + 128 * half);
   // The high-plane load begins at groups 0/2.  A 32-byte shuffle makes
   // [g0,g0,g1,g1] (or [g2,g2,g3,g3]), matching cp=0..3 / cp=4..7.
   const HVX_Vector high = half ? Q6_V_hi_W(high_pair) : Q6_V_lo_W(high_pair);

   HVX_Vector h[4];
   for (unsigned b = 0; b < 4; ++b) {
    const HVX_Vector low_nibble = Q6_Vub_vlsr_VubR(high,b);
    const HVX_Vector high_nibble = Q6_Vub_vlsr_VubR(high,b+4);
    const HVX_Vector selected = Q6_V_vor_VV(Q6_V_vand_VV(high_nibble,shifts),Q6_V_vand_VV(low_nibble,Q6_V_vnot_V(shifts)));
    h[b] = Q6_V_vand_VV(selected, mask1);
    // A byte add is equivalent to <<2 for values known to be 0 or 1.
    h[b] = Q6_Vb_vadd_VbVb(h[b], h[b]);
    h[b] = Q6_Vb_vadd_VbVb(h[b], h[b]);
   }

   HVX_Vector v0 = Q6_V_vor_VV(Q6_V_vand_VV(low, mask2), h[0]);
   HVX_Vector v1 = Q6_V_vor_VV(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(low, 2), mask2), h[1]);
   HVX_Vector v2 = Q6_V_vor_VV(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(low, 4), mask2), h[2]);
   HVX_Vector v3 = Q6_V_vor_VV(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(low, 6), mask2), h[3]);
   v0 = Q6_Vb_vsub_VbVb(v0, four);
   v1 = Q6_Vb_vsub_VbVb(v1, four);
   v2 = Q6_Vb_vsub_VbVb(v2, four);
   v3 = Q6_Vb_vsub_VbVb(v3, four);

   HVX_VectorPair p01 = Q6_W_vshuff_VVR(v1, v0, -1);
   HVX_VectorPair p23 = Q6_W_vshuff_VVR(v3, v2, -1);
   HVX_VectorPair cp01 = Q6_W_vshuff_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -64);
   HVX_VectorPair cp23 = Q6_W_vshuff_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -64);
   HVX_Vector values[4] = { Q6_V_lo_W(cp01), Q6_V_hi_W(cp01),
                            Q6_V_lo_W(cp23), Q6_V_hi_W(cp23) };
   for (unsigned part = 0; part < 4; ++part) {
    HVX_VectorPair unpacked = Q6_Wh_vunpack_Vb(values[part]);
    HVX_Vector a = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(
        Q6_Vhf_equals_Vh(Q6_V_lo_W(unpacked)), scale2));
    HVX_Vector b = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(
        Q6_Vhf_equals_Vh(Q6_V_hi_W(unpacked)), scale2));
    hvx_vmem(out + half * 512 + 128 * part) = a;
    hvx_vmem(out + half * 512 + 128 * part + 64) = b;
   }
  }
 }
}
