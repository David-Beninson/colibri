/* expert_ffn.h — the routed-expert FFN kernel shared by the MoE engines.
 *
 * One weight layout, one activation contract, one way of running a layer:
 *
 *   layout   "planar unsigned int4, gs=64": a row of I weights (I % 64 == 0) is
 *            I/64 blocks of 32 bytes; byte k of a block holds element k in its
 *            low nibble and element k+32 in its high nibble, both stored as
 *            v+8 (0..15). One f32 scale per (row, block), row-major [O][I/64].
 *            The same layout quant.h calls "fmt 4 planar" (K1b): `and 0x0F`
 *            yields elements 0..31 already in order, `srli 4` the other 32,
 *            so no unpack instruction is ever paid, and the unsigned nibbles
 *            feed vpdpbusd natively through dot(v,x) = dot(u,x) - 8*sum(x).
 *   f32      activations stay f32 (default): the weight is exact in f32, one
 *            fma per element, one fma per block for the scale. The only
 *            difference from a pair-layout f32 kernel is accumulation order.
 *   i8       activations quantized once per token to int8 with one scale per
 *            row plus the per-block sums (opt-in, `mode` 1): integer dots,
 *            2 dpbusd per block. Not bit-identical to f32: same policy as
 *            IDOT elsewhere in the tree (opt-in until an ablation blesses it).
 *   layer    xf_moe_run() runs a whole MoE layer: gate+up fused over the
 *            experts the routing touched, silu, down, then the K contributions
 *            of each token summed in rank order. Threads split (expert, row
 *            chunk) items, so a layer costs two OpenMP regions instead of
 *            3*K, and a prompt row never re-reads a weight another row of the
 *            same expert just used.
 *
 * Why it exists: measured on the real Qwen3.6 container at full residency the
 * MoE cost 34 ms/token in steady state against a DRAM floor of 17 ms for the
 * int8-unpacked experts it was reading and 9 ms for the int4 bytes actually
 * on disk. Unpacking int4 to int8 at load doubled both the bytes per token
 * and the resident set; per-GEMV OpenMP regions (24 per layer) and one
 * reduction per 64-block did the rest. The same shape (int4 experts, f32
 * hidden state, top-k routing) is every engine here, which is why this is a
 * header and not a function in one of them.
 *
 * Header-only, all static, pure compute: no Model, no I/O, no allocation
 * beyond what the caller hands in. AVX2/FMA, AVX-VNNI or AVX512-VNNI where built,
 * with a scalar fallback that defines the numerics every SIMD path must
 * reproduce (f32 paths: same per-lane accumulation order; i8 paths: exact
 * integer sums, so identical by construction). */
#ifndef COLI_EXPERT_FFN_H
#define COLI_EXPERT_FFN_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#define XF_BLOCK 64                       /* elements per planar block and per scale group */
#define XF_BLOCK_BYTES 32

/* ---- layout helpers ------------------------------------------------------- */

static inline size_t xf_row_bytes(int I) { return (size_t)I / 2; }
static inline int    xf_groups(int I)    { return I / XF_BLOCK; }
static inline int    xf_layout_ok(int I) { return I > 0 && (I % XF_BLOCK) == 0; }

/* Repack one row from the "signed pairs" container layout (byte j = element
 * 2j in the low nibble, 2j+1 in the high nibble, two's-complement 4-bit — the
 * qwen36 converter's pack_int4) into planar unsigned. src and dst may not
 * overlap. I % 64 == 0. */
static inline void xf_repack_row_pairs_signed(uint8_t *dst, const uint8_t *src, int I) {
    int b = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
#define XF_REPACK_NEON 1   /* this path is compiled in: tests/test_expert_ffn.c checks it on AArch64 */
    /* The same pairing, built directly: byte j of `even` is element 2j in the
     * low nibble and element 32+2j in the high one, `odd` the same for 2j+1,
     * so zipping them byte-wise gives elements k | k+32 in order; xor 0x88 is
     * the xor 8 of both nibbles. Called on every expert read from disk.
     * vzip1q/vzip2q are AArch64 instructions: 32-bit ARM takes the loop below. */
    const uint8x16_t m4 = vdupq_n_u8(0x0F), mh = vdupq_n_u8(0xF0), e8 = vdupq_n_u8(0x88);
    for (; b < I / XF_BLOCK; b++) {
        const uint8_t *s = src + (size_t)b * XF_BLOCK_BYTES;
        uint8_t *d = dst + (size_t)b * XF_BLOCK_BYTES;
        uint8x16_t a = vld1q_u8(s), c = vld1q_u8(s + 16);
        uint8x16_t even = vorrq_u8(vandq_u8(a, m4), vshlq_n_u8(c, 4));
        uint8x16_t odd = vorrq_u8(vshrq_n_u8(a, 4), vandq_u8(c, mh));
        vst1q_u8(d, veorq_u8(vzip1q_u8(even, odd), e8));
        vst1q_u8(d + 16, veorq_u8(vzip2q_u8(even, odd), e8));
    }
#endif
    for (; b < I / XF_BLOCK; b++) {
        const uint8_t *s = src + (size_t)b * XF_BLOCK_BYTES;
        uint8_t *d = dst + (size_t)b * XF_BLOCK_BYTES;
        for (int k = 0; k < 32; k++) {
            /* element k of the block: byte k/2, nibble k&1; element k+32: byte 16 + k/2 */
            unsigned lo = (unsigned)(s[k >> 1]      >> ((k & 1) * 4)) & 0xF;
            unsigned hi = (unsigned)(s[16 + (k >> 1)] >> ((k & 1) * 4)) & 0xF;
            /* signed 4-bit -> v+8 is the nibble xor 8 */
            d[k] = (uint8_t)((lo ^ 8u) | ((hi ^ 8u) << 4));
        }
    }
}
static inline void xf_repack_pairs_signed(uint8_t *dst, const uint8_t *src, int O, int I) {
    for (int o = 0; o < O; o++)
        xf_repack_row_pairs_signed(dst + (size_t)o * xf_row_bytes(I), src + (size_t)o * xf_row_bytes(I), I);
}
/* Decode one element from a planar row (tests and scalar paths). */
static inline int xf_get(const uint8_t *row, int i) {
    const uint8_t *blk = row + (size_t)(i / XF_BLOCK) * XF_BLOCK_BYTES;
    int k = i % XF_BLOCK;
    unsigned u = k < 32 ? (blk[k] & 0xF) : (blk[k - 32] >> 4);
    return (int)u - 8;
}

/* ---- activation contract for the i8 mode ---------------------------------- */

/* Quantize x[I] to int8 with one scale (amax/127), and the per-block sums as
 * f32 (exact: |sum| <= 64*127). Returns the scale. */
static inline float xf_act_i8(const float *x, int I, int8_t *xq, float *xsum) {
    float am = 0.f;
    int i = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
    float32x4_t am0 = vdupq_n_f32(0.f), am1 = vdupq_n_f32(0.f);
    for (; i + 8 <= I; i += 8) {
        am0 = vmaxq_f32(am0, vabsq_f32(vld1q_f32(x + i)));
        am1 = vmaxq_f32(am1, vabsq_f32(vld1q_f32(x + i + 4)));
    }
    am = vmaxvq_f32(vmaxq_f32(am0, am1));
#endif
    for (; i < I; i++) { float a = fabsf(x[i]); if (a > am) am = a; }
    float s = am > 0.f ? am / 127.f : 1.f, inv = 1.f / s;
    i = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
    const float32x4_t vinv = vdupq_n_f32(inv);
    for (; i + 16 <= I; i += 16) {
        int32x4_t a = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i),      vinv));
        int32x4_t b = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 4),  vinv));
        int32x4_t c = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 8),  vinv));
        int32x4_t d = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 12), vinv));
        int16x8_t ab = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
        int16x8_t cd = vcombine_s16(vqmovn_s32(c), vqmovn_s32(d));
        vst1q_s8(xq + i, vcombine_s8(vqmovn_s16(ab), vqmovn_s16(cd)));
    }
#endif
    for (; i < I; i++) xq[i] = (int8_t)lrintf(x[i] * inv);
    for (int g = 0; g < I / XF_BLOCK; g++) {
#if defined(__ARM_NEON) && defined(__aarch64__)
        const int8_t *p = xq + g * XF_BLOCK;
        int32_t t = (int32_t)vaddlvq_s8(vld1q_s8(p))
                  + (int32_t)vaddlvq_s8(vld1q_s8(p + 16))
                  + (int32_t)vaddlvq_s8(vld1q_s8(p + 32))
                  + (int32_t)vaddlvq_s8(vld1q_s8(p + 48));
        xsum[g] = (float)t;
#else
        int32_t t = 0;
        for (int k = 0; k < XF_BLOCK; k++) t += xq[g * XF_BLOCK + k];
        xsum[g] = (float)t;
#endif
    }
    return s;
}

/* ---- scalar reference kernels (the numerics contract) --------------------- */

/* f32: per block, 8 lanes l=0..7 accumulate elements l, l+8, ..., l+56 in
 * order with fma; the 8 lane sums are scaled by the block scale into 8 lane
 * accumulators; the row ends with the fixed 8->1 tree of xf_hsum8. The SIMD
 * paths reproduce exactly this. */
static inline float xf_hsum8_scalar(const float *l) {
    float a0 = l[0] + l[4], a1 = l[1] + l[5], a2 = l[2] + l[6], a3 = l[3] + l[7];
    float b0 = a0 + a2, b1 = a1 + a3;
    return b0 + b1;
}
static inline float xf_dot_f32_ref(const uint8_t *w, const float *sc, const float *x, int I) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int g = 0; g < I / XF_BLOCK; g++) {
        const uint8_t *blk = w + (size_t)g * XF_BLOCK_BYTES;
        const float *xb = x + g * XF_BLOCK;
        float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < XF_BLOCK; i++) {
            int v = (i < 32 ? (blk[i] & 0xF) : (blk[i - 32] >> 4)) - 8;
            lane[i & 7] = fmaf((float)v, xb[i], lane[i & 7]);
        }
        for (int l = 0; l < 8; l++) acc[l] = fmaf(lane[l], sc[g], acc[l]);
    }
    return xf_hsum8_scalar(acc);
}
/* i8: y = sx * sum_g sc[g] * (dot_u(g) - 8*xsum[g]); the per-block integer
 * dot is exact, then one f32 fma per block per lane exactly as the f32 path
 * (the 8 lanes hold the 8 int32 partial sums vpdpbusd produces: lane l sums
 * elements 4l..4l+3 and 32+4l..32+4l+3 of the block). */
static inline float xf_dot_i8_ref(const uint8_t *w, const float *sc, const int8_t *xq, const float *xsum, float sx, int I) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float corr = 0.f;
    for (int g = 0; g < I / XF_BLOCK; g++) {
        const uint8_t *blk = w + (size_t)g * XF_BLOCK_BYTES;
        const int8_t *xb = xq + g * XF_BLOCK;
        int32_t lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int k = 0; k < 32; k++) {
            lane[k >> 2] += (int32_t)(blk[k] & 0xF) * xb[k];
            lane[k >> 2] += (int32_t)(blk[k] >> 4)  * xb[k + 32];
        }
        for (int l = 0; l < 8; l++) acc[l] = fmaf((float)lane[l], sc[g], acc[l]);
        corr = fmaf(sc[g], xsum[g], corr);
    }
    return (xf_hsum8_scalar(acc) - 8.f * corr) * sx;
}

/* ---- SIMD kernels ---------------------------------------------------------- */

#if defined(__ARM_NEON) && defined(__aarch64__)
#define XF_HAVE_NEON 1
static inline float xf_hsum8_neon(float32x4_t acc0, float32x4_t acc1) {
    float32x4_t a = vaddq_f32(acc0, acc1);                /* l + (l+4) */
    float32x2_t b = vadd_f32(vget_low_f32(a), vget_high_f32(a)); /* (0+4)+(2+6), (1+5)+(3+7) */
    return vget_lane_f32(vpadd_f32(b, b), 0);
}
static inline void xf_widen16_s8_f32(int8x16_t v, float32x4_t *f0, float32x4_t *f1, float32x4_t *f2, float32x4_t *f3) {
    int16x8_t wlo = vmovl_s8(vget_low_s8(v)), whi = vmovl_s8(vget_high_s8(v));
    *f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wlo)));
    *f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wlo)));
    *f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(whi)));
    *f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(whi)));
}
static inline float xf_dot_f32_neon(const uint8_t *w, const float *sc, const float *x, int I) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int8x16_t e8 = vdupq_n_s8(8);
    float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
    for (int g = 0; g < I / XF_BLOCK; g++) {
        const uint8_t *blk = w + (size_t)g * XF_BLOCK_BYTES;
        uint8x16_t b0 = vld1q_u8(blk), b1 = vld1q_u8(blk + 16);
        int8x16_t lo0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b0, m4)), e8);
        int8x16_t lo1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b1, m4)), e8);
        int8x16_t hi0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b0, 4)), e8);
        int8x16_t hi1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b1, 4)), e8);
        const float *xb = x + g * XF_BLOCK;
        float32x4_t lane0 = vdupq_n_f32(0.f), lane1 = vdupq_n_f32(0.f);
        float32x4_t f0, f1, f2, f3;
        xf_widen16_s8_f32(lo0, &f0, &f1, &f2, &f3);
        lane0 = vfmaq_f32(lane0, f0, vld1q_f32(xb));      lane1 = vfmaq_f32(lane1, f1, vld1q_f32(xb + 4));
        lane0 = vfmaq_f32(lane0, f2, vld1q_f32(xb + 8));  lane1 = vfmaq_f32(lane1, f3, vld1q_f32(xb + 12));
        xf_widen16_s8_f32(lo1, &f0, &f1, &f2, &f3);
        lane0 = vfmaq_f32(lane0, f0, vld1q_f32(xb + 16)); lane1 = vfmaq_f32(lane1, f1, vld1q_f32(xb + 20));
        lane0 = vfmaq_f32(lane0, f2, vld1q_f32(xb + 24)); lane1 = vfmaq_f32(lane1, f3, vld1q_f32(xb + 28));
        xf_widen16_s8_f32(hi0, &f0, &f1, &f2, &f3);
        lane0 = vfmaq_f32(lane0, f0, vld1q_f32(xb + 32)); lane1 = vfmaq_f32(lane1, f1, vld1q_f32(xb + 36));
        lane0 = vfmaq_f32(lane0, f2, vld1q_f32(xb + 40)); lane1 = vfmaq_f32(lane1, f3, vld1q_f32(xb + 44));
        xf_widen16_s8_f32(hi1, &f0, &f1, &f2, &f3);
        lane0 = vfmaq_f32(lane0, f0, vld1q_f32(xb + 48)); lane1 = vfmaq_f32(lane1, f1, vld1q_f32(xb + 52));
        lane0 = vfmaq_f32(lane0, f2, vld1q_f32(xb + 56)); lane1 = vfmaq_f32(lane1, f3, vld1q_f32(xb + 60));
        float32x4_t scv = vdupq_n_f32(sc[g]);
        acc0 = vfmaq_f32(acc0, lane0, scv);
        acc1 = vfmaq_f32(acc1, lane1, scv);
    }
    return xf_hsum8_neon(acc0, acc1);
}

#if defined(__ARM_FEATURE_DOTPROD)
static inline float xf_dot_i8_neon(const uint8_t *w, const float *sc, const int8_t *xq, const float *xsum, float sx, int I) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int32x4_t z = vdupq_n_s32(0);
    float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
    int ng = I / XF_BLOCK;
    for (int g = 0; g < ng; g++) {
        const uint8_t *blk = w + (size_t)g * XF_BLOCK_BYTES;
        const int8_t *xb = xq + g * XF_BLOCK;
        uint8x16_t b0 = vld1q_u8(blk), b1 = vld1q_u8(blk + 16);
        int32x4_t d0 = vdotq_s32(z, vreinterpretq_s8_u8(vandq_u8(b0, m4)), vld1q_s8(xb));
        int32x4_t d1 = vdotq_s32(z, vreinterpretq_s8_u8(vandq_u8(b1, m4)), vld1q_s8(xb + 16));
        d0 = vdotq_s32(d0, vreinterpretq_s8_u8(vshrq_n_u8(b0, 4)), vld1q_s8(xb + 32));
        d1 = vdotq_s32(d1, vreinterpretq_s8_u8(vshrq_n_u8(b1, 4)), vld1q_s8(xb + 48));
        float32x4_t scv = vdupq_n_f32(sc[g]);
        acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(d0), scv);
        acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(d1), scv);
    }
    float c = 0.f;
    for (int g = 0; g < ng; g++) c = fmaf(sc[g], xsum[g], c);
    return (xf_hsum8_neon(acc0, acc1) - 8.f * c) * sx;
}
#endif
#endif /* XF_HAVE_NEON */

/* ---- dispatch -------------------------------------------------------------- */

static inline float xf_dot_f32(const uint8_t *w, const float *sc, const float *x, int I) {
#ifdef XF_HAVE_NEON
    return xf_dot_f32_neon(w, sc, x, I);
#else
    return xf_dot_f32_ref(w, sc, x, I);
#endif
}
#define XF_F32_ROWS 4
#ifdef XF_HAVE_NEON
static inline __attribute__((always_inline)) void xf_dot_f32_rows_body(float *y, int ystride, const uint8_t *w, const float *sc,
                                                                      const float *const *x, int I, const int S) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int8x16_t e8 = vdupq_n_s8(8);
    float32x4_t acc0[XF_F32_ROWS], acc1[XF_F32_ROWS];
    for (int s = 0; s < S; s++) { acc0[s] = vdupq_n_f32(0.f); acc1[s] = vdupq_n_f32(0.f); }
    for (int g = 0; g < I / XF_BLOCK; g++) {
        const uint8_t *blk = w + (size_t)g * XF_BLOCK_BYTES;
        uint8x16_t b0 = vld1q_u8(blk), b1 = vld1q_u8(blk + 16);
        int8x16_t lo0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b0, m4)), e8);
        int8x16_t lo1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b1, m4)), e8);
        int8x16_t hi0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b0, 4)), e8);
        int8x16_t hi1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b1, 4)), e8);
        float32x4_t wf[16];
        xf_widen16_s8_f32(lo0, &wf[0],  &wf[1],  &wf[2],  &wf[3]);
        xf_widen16_s8_f32(lo1, &wf[4],  &wf[5],  &wf[6],  &wf[7]);
        xf_widen16_s8_f32(hi0, &wf[8],  &wf[9],  &wf[10], &wf[11]);
        xf_widen16_s8_f32(hi1, &wf[12], &wf[13], &wf[14], &wf[15]);
        const float32x4_t scv = vdupq_n_f32(sc[g]);
        for (int s = 0; s < S; s++) {
            const float *xb = x[s] + g * XF_BLOCK;
            float32x4_t l0 = vdupq_n_f32(0.f), l1 = vdupq_n_f32(0.f);
            for (int k = 0; k < 8; k++) {
                l0 = vfmaq_f32(l0, wf[2 * k],     vld1q_f32(xb + 8 * k));
                l1 = vfmaq_f32(l1, wf[2 * k + 1], vld1q_f32(xb + 8 * k + 4));
            }
            acc0[s] = vfmaq_f32(acc0[s], l0, scv);
            acc1[s] = vfmaq_f32(acc1[s], l1, scv);
        }
    }
    for (int s = 0; s < S; s++) y[(size_t)s * ystride] = xf_hsum8_neon(acc0[s], acc1[s]);
}
#endif
static inline void xf_dot_f32_rows(float *y, int ystride, const uint8_t *w, const float *sc, const float *const *x, int I, int S) {
    for (int s0 = 0; s0 < S; s0 += XF_F32_ROWS) {
        const int n = S - s0 < XF_F32_ROWS ? S - s0 : XF_F32_ROWS;
#ifdef XF_HAVE_NEON
        switch (n) {
        case 1: xf_dot_f32_rows_body(y + (size_t)s0 * ystride, ystride, w, sc, x + s0, I, 1); break;
        case 2: xf_dot_f32_rows_body(y + (size_t)s0 * ystride, ystride, w, sc, x + s0, I, 2); break;
        case 3: xf_dot_f32_rows_body(y + (size_t)s0 * ystride, ystride, w, sc, x + s0, I, 3); break;
        default: xf_dot_f32_rows_body(y + (size_t)s0 * ystride, ystride, w, sc, x + s0, I, 4); break;
        }
#else
        for (int s = 0; s < n; s++) y[(size_t)(s0 + s) * ystride] = xf_dot_f32_ref(w, sc, x[s0 + s], I);
#endif
    }
}
static inline float xf_dot_i8(const uint8_t *w, const float *sc, const int8_t *xq, const float *xsum, float sx, int I) {
#if defined(XF_HAVE_NEON) && defined(__ARM_FEATURE_DOTPROD)
    return xf_dot_i8_neon(w, sc, xq, xsum, sx, I);
#else
    return xf_dot_i8_ref(w, sc, xq, xsum, sx, I);
#endif
}

#define XF_MAX_ROWS 4
#if defined(XF_HAVE_NEON) && defined(__ARM_FEATURE_DOTPROD)
static inline __attribute__((always_inline)) void xf_dot_i8_rows_body(float *y, int ystride, const uint8_t *w, const float *sc,
                                                                      const int8_t *xq, const float *xsum, const float *sx, int I, const int n) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int32x4_t z = vdupq_n_s32(0);
    float32x4_t acc0[XF_MAX_ROWS], acc1[XF_MAX_ROWS];
    for (int r = 0; r < n; r++) { acc0[r] = vdupq_n_f32(0.f); acc1[r] = vdupq_n_f32(0.f); }
    const int ng = I / XF_BLOCK;
    for (int g = 0; g < ng; g++) {
        const uint8_t *blk = w + (size_t)g * XF_BLOCK_BYTES;
        uint8x16_t b0 = vld1q_u8(blk), b1 = vld1q_u8(blk + 16);
        int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(b0, m4)), lo1 = vreinterpretq_s8_u8(vandq_u8(b1, m4));
        int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(b0, 4)), hi1 = vreinterpretq_s8_u8(vshrq_n_u8(b1, 4));
        float32x4_t scg = vdupq_n_f32(sc[g]);
        for (int r = 0; r < n; r++) {
            const int8_t *xb = xq + (size_t)r * I + g * XF_BLOCK;
            int32x4_t d0 = vdotq_s32(z, lo0, vld1q_s8(xb));
            int32x4_t d1 = vdotq_s32(z, lo1, vld1q_s8(xb + 16));
            d0 = vdotq_s32(d0, hi0, vld1q_s8(xb + 32));
            d1 = vdotq_s32(d1, hi1, vld1q_s8(xb + 48));
            acc0[r] = vfmaq_f32(acc0[r], vcvtq_f32_s32(d0), scg);
            acc1[r] = vfmaq_f32(acc1[r], vcvtq_f32_s32(d1), scg);
        }
    }
    for (int r = 0; r < n; r++) {
        const float *xs = xsum + (size_t)r * ng;
        float c = 0.f;
        for (int g = 0; g < ng; g++) c = fmaf(sc[g], xs[g], c);
        y[(size_t)r * ystride] = (xf_hsum8_neon(acc0[r], acc1[r]) - 8.f * c) * sx[r];
    }
}
#endif
static inline void xf_dot_i8_rows(float *y, int ystride, const uint8_t *w, const float *sc, const int8_t *xq,
                                  const float *xsum, const float *sx, int I, int n) {
#if defined(XF_HAVE_NEON) && defined(__ARM_FEATURE_DOTPROD)
    switch (n) {
    case 1: xf_dot_i8_rows_body(y, ystride, w, sc, xq, xsum, sx, I, 1); break;
    case 2: xf_dot_i8_rows_body(y, ystride, w, sc, xq, xsum, sx, I, 2); break;
    case 3: xf_dot_i8_rows_body(y, ystride, w, sc, xq, xsum, sx, I, 3); break;
    default: xf_dot_i8_rows_body(y, ystride, w, sc, xq, xsum, sx, I, 4); break;
    }
#else
    for (int r = 0; r < n; r++)
        y[(size_t)r * ystride] = xf_dot_i8_ref(w, sc, xq + (size_t)r * I, xsum + (size_t)r * (I / XF_BLOCK), sx[r], I);
#endif
}

/* ---- rows interleaved by 16 (xf16_*) ------------------------------------------
 * The row kernels above keep a float lane sum per row, one fma per element (f32) or
 * one per block (i8), and pay the nibble widening once per activation row: a
 * prompt's rows cost as many passes as there are rows. Here 16 rows share a
 * register (lane r = row r). A group of 16 rows holds, per 64 inputs (block b),
 *   64 bytes   the 16 rows' f32 scales of b
 *   512 bytes  words W = 0..7 of 64 bytes, row r's four bytes at 4r: bytes 4W..4W+3
 *              of its planar block (low nibbles elements 4W..4W+3, high 32+4W..)
 * so quad W of the 16 rows is one mask and quad 8+W a shift and a mask, each one
 * vpdpbusd against the activation's broadcast quad. The same bytes as the rows
 * and their scales.
 * Activations: int8 per 32 with a scale per 32 and the sums per 16 (xf_act32: the
 * GGUF kernels' gq_quant_act, byte for byte), not the i8 mode's one per row.
 * Numerics: per 32 inputs the exact integer sum (u - 8) x, p0 and p1 for the two
 * halves of a block, then y = fma(fma(p1, dx1, p0 * dx0), scale, y) block by
 * block: xf16_ref, the bits of every SIMD path whatever S. */
#define XF16_BLOCK_BYTES 576
static inline size_t xf16_group_bytes(int I) { return (size_t)(I / XF_BLOCK) * XF16_BLOCK_BYTES; }
/* rows (whole groups) of planar codes w [rows][I/2] and scales sc [rows][I/64] into dst */
static void xf16_pack(uint8_t *dst, const uint8_t *w, const float *sc, int rows, int I) {
    const size_t rb = xf_row_bytes(I); const int nb = I / XF_BLOCK;
    for (int g = 0; g < rows / 16; g++)
        for (int b = 0; b < nb; b++) {
            uint8_t *o = dst + ((size_t)g * nb + b) * XF16_BLOCK_BYTES;
            for (int r = 0; r < 16; r++) {
                const uint8_t *blk = w + (size_t)(16 * g + r) * rb + (size_t)b * XF_BLOCK_BYTES;
                memcpy(o + 4 * r, sc + (size_t)(16 * g + r) * nb + b, 4);
                for (int W = 0; W < 8; W++) memcpy(o + 64 + 64 * W + 4 * r, blk + 4 * W, 4);
            }
        }
}
/* the inverse: planar codes and scales back from the groups */
static void xf16_unpack(uint8_t *w, float *sc, const uint8_t *src, int rows, int I) {
    const size_t rb = xf_row_bytes(I); const int nb = I / XF_BLOCK;
    for (int g = 0; g < rows / 16; g++)
        for (int b = 0; b < nb; b++) {
            const uint8_t *o = src + ((size_t)g * nb + b) * XF16_BLOCK_BYTES;
            for (int r = 0; r < 16; r++) {
                uint8_t *blk = w + (size_t)(16 * g + r) * rb + (size_t)b * XF_BLOCK_BYTES;
                memcpy(sc + (size_t)(16 * g + r) * nb + b, o + 4 * r, 4);
                for (int W = 0; W < 8; W++) memcpy(blk + 4 * W, o + 64 + 64 * W + 4 * r, 4);
            }
        }
}
/* x[I] as int8 per 32: scale amax/127 per 32 (dx), the codes rounded as lrintf and
 * clamped to +-127, the sums of every 16 codes (xs); ggml_quants.h's gq_quant_act */
static inline void xf_act32_ref(const float *x, int I, int8_t *xq, float *dx, int32_t *xs) {
    for (int b = 0; b < I / 32; b++) {
        const float *v = x + b * 32; float amax = 0.f;
        for (int i = 0; i < 32; i++) { float a = fabsf(v[i]); if (a > amax) amax = a; }
        const float id = amax > 0.f ? 127.f / amax : 0.f;
        dx[b] = amax / 127.f;
        int32_t s0 = 0, s1 = 0;
        for (int i = 0; i < 32; i++) {
            int q = (int)lrintf(v[i] * id); if (q > 127) q = 127; if (q < -127) q = -127;
            xq[b * 32 + i] = (int8_t)q; if (i < 16) s0 += q; else s1 += q;
        }
        xs[2 * b] = s0; xs[2 * b + 1] = s1;
    }
}
/* row r of a group (rows at w, scales at sc as xf16_pack takes them) against one row */
static inline float xf16_ref(const uint8_t *w, const float *sc, const int8_t *xq, const float *dx, int I) {
    float acc = 0.f;
    for (int b = 0; b < I / XF_BLOCK; b++) {
        const uint8_t *blk = w + (size_t)b * XF_BLOCK_BYTES; const int8_t *xb = xq + (size_t)b * XF_BLOCK;
        int32_t p0 = 0, p1 = 0;
        for (int k = 0; k < 32; k++) { p0 += ((int)(blk[k] & 15) - 8) * xb[k]; p1 += ((int)(blk[k] >> 4) - 8) * xb[32 + k]; }
        acc = fmaf(fmaf((float)p1, dx[2 * b + 1], (float)p0 * dx[2 * b]), sc[b], acc);
    }
    return acc;
}
#if defined(__ARM_NEON) && defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
#define XF16_ON 1   /* the interleaved form has kernels in this build: the engines may make it */
static inline int32_t xf16_ld32(const int8_t *p) { int32_t v; memcpy(&v, p, 4); return v; }
static inline void xf_act32_block(const float *v, int8_t *q, float *d, int32_t *s16) {
    const float32x4_t v0 = vld1q_f32(v),      v1 = vld1q_f32(v + 4),  v2 = vld1q_f32(v + 8),  v3 = vld1q_f32(v + 12);
    const float32x4_t v4 = vld1q_f32(v + 16), v5 = vld1q_f32(v + 20), v6 = vld1q_f32(v + 24), v7 = vld1q_f32(v + 28);
    const float32x4_t a0 = vmaxq_f32(vmaxq_f32(vabsq_f32(v0), vabsq_f32(v1)), vmaxq_f32(vabsq_f32(v2), vabsq_f32(v3)));
    const float32x4_t a1 = vmaxq_f32(vmaxq_f32(vabsq_f32(v4), vabsq_f32(v5)), vmaxq_f32(vabsq_f32(v6), vabsq_f32(v7)));
    const float amax = vmaxvq_f32(vmaxq_f32(a0, a1));
    const float id = amax > 0.f ? 127.f / amax : 0.f;
    *d = amax / 127.f;
    const float32x4_t idv = vdupq_n_f32(id);
    const int32x4_t lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
    const int32x4_t q0 = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(vmulq_f32(v0, idv)), lo), hi);
    const int32x4_t q1 = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(vmulq_f32(v1, idv)), lo), hi);
    const int32x4_t q2 = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(vmulq_f32(v2, idv)), lo), hi);
    const int32x4_t q3 = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(vmulq_f32(v3, idv)), lo), hi);
    const int32x4_t q4 = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(vmulq_f32(v4, idv)), lo), hi);
    const int32x4_t q5 = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(vmulq_f32(v5, idv)), lo), hi);
    const int32x4_t q6 = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(vmulq_f32(v6, idv)), lo), hi);
    const int32x4_t q7 = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(vmulq_f32(v7, idv)), lo), hi);
    vst1q_s8(q,      vcombine_s8(vqmovn_s16(vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1))),
                                 vqmovn_s16(vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3)))));
    vst1q_s8(q + 16, vcombine_s8(vqmovn_s16(vcombine_s16(vqmovn_s32(q4), vqmovn_s32(q5))),
                                 vqmovn_s16(vcombine_s16(vqmovn_s32(q6), vqmovn_s32(q7)))));
    s16[0] = vaddvq_s32(vaddq_s32(vaddq_s32(q0, q1), vaddq_s32(q2, q3)));
    s16[1] = vaddvq_s32(vaddq_s32(vaddq_s32(q4, q5), vaddq_s32(q6, q7)));
}
static inline void xf_act32(const float *x, int I, int8_t *xq, float *dx, int32_t *xs) {
    for (int b = 0; b < I / 32; b++) xf_act32_block(x + b * 32, xq + b * 32, dx + b, xs + 2 * b);
}
static inline float xf_dot_a32(const uint8_t *w, const float *sc, const int8_t *xq, const float *dx, const int32_t *xs, int I) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int32x4_t z = vdupq_n_s32(0);
    const int nb = I / XF_BLOCK;
    float acc = 0.f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = w + (size_t)b * XF_BLOCK_BYTES;
        const int8_t *xb = xq + (size_t)b * XF_BLOCK;
        uint8x16_t b0 = vld1q_u8(blk), b1 = vld1q_u8(blk + 16);
        int32x4_t v0 = vdotq_s32(z,  vreinterpretq_s8_u8(vandq_u8(b0, m4)), vld1q_s8(xb));
        v0 = vdotq_s32(v0, vreinterpretq_s8_u8(vandq_u8(b1, m4)), vld1q_s8(xb + 16));
        int32x4_t v1 = vdotq_s32(z,  vreinterpretq_s8_u8(vshrq_n_u8(b0, 4)), vld1q_s8(xb + 32));
        v1 = vdotq_s32(v1, vreinterpretq_s8_u8(vshrq_n_u8(b1, 4)), vld1q_s8(xb + 48));
        const int32_t *x16 = xs + 4 * (size_t)b;
        const int32_t p0 = vaddvq_s32(v0) - 8 * (x16[0] + x16[1]);
        const int32_t p1 = vaddvq_s32(v1) - 8 * (x16[2] + x16[3]);
        acc = fmaf(fmaf((float)p1, dx[2 * b + 1], (float)p0 * dx[2 * b]), sc[b], acc);
    }
    return acc;
}
/* quarter h (rows 4h .. 4h+3) of a group against S <= 4 activation rows */
static inline __attribute__((always_inline)) void xf16_body128(float *const *y, int o, const uint8_t *grp, const int8_t *const *xq,
                                                              const float *const *dx, const int32_t *const *xs, int I, const int S, const int h) {
    const uint8x16_t m4 = vdupq_n_u8(15);
    float32x4_t acc[4];
    for (int s = 0; s < S; s++) acc[s] = vdupq_n_f32(0.f);
    for (int b = 0; b < I / XF_BLOCK; b++) {
        const uint8_t *g = grp + (size_t)b * XF16_BLOCK_BYTES;
        const float32x4_t sv = vld1q_f32((const float *)g + 4 * h);
        int8x16_t lo[8], hi[8];
        for (int W = 0; W < 8; W++) {
            const uint8x16_t v = vld1q_u8(g + 64 + 64 * W + 16 * h);
            lo[W] = vreinterpretq_s8_u8(vandq_u8(v, m4));
            hi[W] = vreinterpretq_s8_u8(vshrq_n_u8(v, 4));
        }
        for (int s = 0; s < S; s++) {
            const int8_t *xb = xq[s] + (size_t)b * XF_BLOCK;
            int32x4_t p0 = vdupq_n_s32(0), p1 = vdupq_n_s32(0);
            for (int W = 0; W < 8; W++) {
                p0 = vdotq_s32(p0, lo[W], vreinterpretq_s8_s32(vdupq_n_s32(xf16_ld32(xb + 4 * W))));
                p1 = vdotq_s32(p1, hi[W], vreinterpretq_s8_s32(vdupq_n_s32(xf16_ld32(xb + 32 + 4 * W))));
            }
            const int32_t *x16 = xs[s] + 4 * (size_t)b; const float *d = dx[s] + 2 * (size_t)b;
            p0 = vsubq_s32(p0, vdupq_n_s32(8 * (x16[0] + x16[1])));
            p1 = vsubq_s32(p1, vdupq_n_s32(8 * (x16[2] + x16[3])));
            const float32x4_t t = vfmaq_f32(vmulq_f32(vcvtq_f32_s32(p0), vdupq_n_f32(d[0])),
                                            vcvtq_f32_s32(p1), vdupq_n_f32(d[1]));
            acc[s] = vfmaq_f32(acc[s], t, sv);
        }
    }
    for (int s = 0; s < S; s++) vst1q_f32(y[s] + o + 4 * h, acc[s]);
}
#define XF16_SMAX 4
static void xf16_rows(float *const *y, int o, const uint8_t *grp, const int8_t *const *xq, const float *const *dx,
                      const int32_t *const *xs, int I, int S) {
    for (int s0 = 0; s0 < S; s0 += XF16_SMAX) {
        const int n = S - s0 < XF16_SMAX ? S - s0 : XF16_SMAX;
#define XF16_CALL(N) do { \
    xf16_body128(y + s0, o, grp, xq + s0, dx + s0, xs + s0, I, N, 0); \
    xf16_body128(y + s0, o, grp, xq + s0, dx + s0, xs + s0, I, N, 1); \
    xf16_body128(y + s0, o, grp, xq + s0, dx + s0, xs + s0, I, N, 2); \
    xf16_body128(y + s0, o, grp, xq + s0, dx + s0, xs + s0, I, N, 3); \
} while (0)
        switch (n) { case 1: XF16_CALL(1); break; case 2: XF16_CALL(2); break; case 3: XF16_CALL(3); break;
                     case 4: XF16_CALL(4); break; default: XF16_CALL(n); break; }
#undef XF16_CALL
    }
}
#endif /* XF16_ON */

/* ---- one expert ------------------------------------------------------------- */

typedef struct {
    const uint8_t *g4, *u4, *d4;   /* planar rows: gate/up [F][H], down [H][F] */
    const float *gs, *us, *ds;     /* block scales: [F][H/64], [F][H/64], [H][F/64] */
} XfExpert;

/* Activations of one token, prepared once per layer (both modes carry x). */
typedef struct {
    const float *x;      /* [H] f32 */
    const int8_t *xq;    /* [H] int8, i8 mode */
    const float *xsum;   /* [H/64] block sums, i8 mode */
    float sx;            /* activation scale, i8 mode */
} XfAct;

/* gate+up rows [r0,r1) of one expert for one token: g[r], u[r]. Both matrices
 * share the activation, so their rows are interleaved in the same pass. */
static inline void xf_gate_up_rows(float *g, float *u, const XfExpert *e, const XfAct *a, int H, int r0, int r1, int mode) {
    size_t rb = xf_row_bytes(H); int ng = xf_groups(H);
    for (int r = r0; r < r1; r++) {
        const uint8_t *wg = e->g4 + (size_t)r * rb, *wu = e->u4 + (size_t)r * rb;
        const float *sg = e->gs + (size_t)r * ng, *su = e->us + (size_t)r * ng;
        if (mode) { g[r] = xf_dot_i8(wg, sg, a->xq, a->xsum, a->sx, H); u[r] = xf_dot_i8(wu, su, a->xq, a->xsum, a->sx, H); }
        else      { g[r] = xf_dot_f32(wg, sg, a->x, H);                 u[r] = xf_dot_f32(wu, su, a->x, H); }
    }
}
/* silu(g)*u over [0,F): the hidden row, in the engines' historical form */
static inline void xf_swiglu(float *h, const float *g, const float *u, int F) {
    for (int i = 0; i < F; i++) { float gv = g[i]; h[i] = (gv / (1.f + expf(-gv))) * u[i]; }
}
/* down rows [r0,r1) of one expert for one hidden row */
static inline void xf_down_rows(float *y, const XfExpert *e, const XfAct *h, int F, int r0, int r1, int mode) {
    size_t rb = xf_row_bytes(F); int ng = xf_groups(F);
    for (int r = r0; r < r1; r++) {
        const uint8_t *wd = e->d4 + (size_t)r * rb; const float *sd = e->ds + (size_t)r * ng;
        y[r] = mode ? xf_dot_i8(wd, sd, h->xq, h->xsum, h->sx, F) : xf_dot_f32(wd, sd, h->x, F);
    }
}

/* ---- one layer ---------------------------------------------------------------
 *
 * S tokens, K routed experts each. idx[s*K+k] is the expert id (or -1), val the
 * router weight, experts[s*K+k] the resident expert (NULL iff idx < 0).
 * out[s][H] = sum_k val[s][k] * expert_k(x[s]) accumulated in k order, as the
 * per-token loops it replaces did. xf_moe_add adds that sum into out instead of
 * overwriting it: a caller that cuts one layer into several runs (qwen36, when
 * its cache holds fewer experts than the batch routes to) sends every cut
 * through the same statement, so the bits cannot depend on where it cut, whether
 * or not the compiler fuses that multiply-add. xf_moe_run zeroes out first.
 * Callers own `scratch` (xf_moe_scratch_bytes).
 *
 * Work is split into (expert, row chunk) items twice: gate+up over F rows, then
 * down over H rows. A prompt of S rows routed to the same expert reads that
 * expert's weights from DRAM once per chunk, the other rows hit cache. */
static inline size_t xf_moe_scratch_bytes(int S, int K, int H, int F) {
    size_t n = (size_t)S * K;
    return n * (2 * (size_t)F + (size_t)H) * sizeof(float)      /* g,u per (s,k); contrib per (s,k) */
         + n * (size_t)F * sizeof(float)                           /* h per (s,k) */
         + n * ((size_t)F + 64) + n * (size_t)(F / XF_BLOCK) * sizeof(float) /* hq, hsum (i8 mode) */
         + (size_t)S * ((size_t)H + 64) + (size_t)S * (size_t)(H / XF_BLOCK) * sizeof(float) /* xq, xsum */
         + n * sizeof(int) * 4 + 4096
         + (n * (size_t)(F / 32 + F / 16) + (size_t)S * (H / 32 + H / 16)) * 4 + n * sizeof(int) * 3 + 1024;   /* XF_MODE_R16 */
}

/* How many rows from s0 one run can take when the caller's cache holds `cap`
 * experts at once: the longest stretch whose distinct experts number at most
 * cap, at most `max` rows, never fewer than one (a caller whose cap is below K
 * cuts that row into pairs itself). The rows of a run that share an expert read
 * it once per chunk, so the cut decides how often a prompt reads its experts;
 * the bits do not depend on it (xf_moe_add). An id of XF_FIT_IDS or more ends
 * the run before its row. */
#define XF_FIT_IDS 1024
static inline int xf_rows_fit(const int *idx, int s0, int S, int K, int cap, int max) {
    uint8_t seen[XF_FIT_IDS] = {0};
    int distinct = 0, rows = 0;
    for (; s0 + rows < S && rows < max; rows++) {
        const int *r = idx + (size_t)(s0 + rows) * K;
        int add = 0;
        for (int k = 0; k < K; k++) {
            int e = r[k];
            if (e < 0) continue;
            if (e >= XF_FIT_IDS) return rows > 0 ? rows : 1;
            if (!seen[e]) { seen[e] = 1; add++; }
        }
        if (distinct + add > cap) break;
        distinct += add;
    }
    return rows > 0 ? rows : 1;
}

/* mode XF_MODE_R16: the experts' matrices in xf16 groups (XfExpert g4/u4/d4 the groups,
 * the scales inside them), the activations per 32 (xf_act32). XF_MODE_A32: the same
 * numbers on planar experts (xf_dot_a32), for slots a device reads as rows. Made only
 * where XF16_ON. */
#define XF_MODE_R16 2
#define XF_MODE_A32 3
#ifdef XF16_ON
/* xf_moe_compute's work in that mode: the same (expert, row chunk) items, rows by 16, each
 * expert's routed rows handed to the kernel as pointers (no gather) */
static inline float *xf_moe_compute_r16(float *ctb_out, const float *x, int S, int K, int H, int F,
                                        const int *idx, const XfExpert *const *experts, void *scratch, int planar) {
    const size_t n = (size_t)S * K;
    char *p = (char *)scratch;
#define XF_TAKE(T, count) ((T *)p); p += (((size_t)(count) * sizeof(T)) + 63) & ~(size_t)63
    float *g     = XF_TAKE(float, n * F);
    float *u     = XF_TAKE(float, n * F);
    float *h     = XF_TAKE(float, n * F);
    float *ctb   = XF_TAKE(float, n * H);
    int8_t *hq   = XF_TAKE(int8_t, n * F);
    float *hdx   = XF_TAKE(float, n * (F / 32));
    int32_t *hxs = XF_TAKE(int32_t, n * (F / 16));
    int8_t *xq   = XF_TAKE(int8_t, (size_t)S * H);
    float *xdx   = XF_TAKE(float, (size_t)S * (H / 32));
    int32_t *xxs = XF_TAKE(int32_t, (size_t)S * (H / 16));
    int *uniq    = XF_TAKE(int, n);          /* distinct expert ids, in first-seen order */
    int *head    = XF_TAKE(int, n);
    int *next    = XF_TAKE(int, n);
    int *start   = XF_TAKE(int, n + 1);      /* expert j's pairs: ord[start[j] .. start[j+1]) */
    int *ord     = XF_TAKE(int, n);
#undef XF_TAKE
    if (ctb_out) ctb = ctb_out;

    #pragma omp parallel for schedule(static) if (S >= 8)
    for (int s = 0; s < S; s++) xf_act32(x + (size_t)s * H, H, xq + (size_t)s * H, xdx + (size_t)s * (H / 32), xxs + (size_t)s * (H / 16));

    int nu = 0;
    for (int s = 0; s < S; s++) for (int k = 0; k < K; k++) {
        int i = s * K + k; next[i] = -1;
        if (idx[i] < 0 || !experts[i]) continue;
        int j = 0; for (; j < nu; j++) if (uniq[j] == idx[i]) break;
        if (j == nu) { uniq[nu] = idx[i]; head[nu] = i; nu++; }
        else { int t = head[j]; while (next[t] >= 0) t = next[t]; next[t] = i; }
    }
    if (nu == 0) return NULL;
    start[0] = 0;
    for (int j = 0; j < nu; j++) { int c = start[j]; for (int i = head[j]; i >= 0; i = next[i]) ord[c++] = i; start[j + 1] = c; }

    int T = 1;
#ifdef _OPENMP
    T = omp_get_max_threads();
#endif
    /* chunking as xf_moe_compute's, rows in multiples of 16 */
    int cF = (4 * T + nu - 1) / nu; if (cF < 1) cF = 1; if (cF > F / 16) cF = F / 16 > 0 ? F / 16 : 1;
    int cH = (4 * T + nu - 1) / nu; if (cH < 1) cH = 1; if (cH > H / 16) cH = H / 16 > 0 ? H / 16 : 1;
    int rowsF = (F + cF - 1) / cF; rowsF = (rowsF + 15) & ~15; cF = (F + rowsF - 1) / rowsF;
    int rowsH = (H + cH - 1) / cH; rowsH = (rowsH + 15) & ~15; cH = (H + rowsH - 1) / rowsH;
    const size_t gbH = xf16_group_bytes(H), gbF = xf16_group_bytes(F);
#define XF16_PAIRS 64
    /* phase 1: gate+up for every (expert, chunk), all rows routed to that expert */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int it = 0; it < nu * cF; it++) {
        int e = it / cF, c = it % cF;
        int r0 = c * rowsF, r1 = r0 + rowsF; if (r1 > F) r1 = F;
        const XfExpert *ex = experts[ord[start[e]]];
        for (int j0 = start[e]; j0 < start[e + 1]; j0 += XF16_PAIRS) {
            const int m = start[e + 1] - j0 < XF16_PAIRS ? start[e + 1] - j0 : XF16_PAIRS;
            float *yg[XF16_PAIRS], *yu[XF16_PAIRS]; const int8_t *xp[XF16_PAIRS]; const float *dp[XF16_PAIRS]; const int32_t *sp[XF16_PAIRS];
            for (int j = 0; j < m; j++) {
                const int i = ord[j0 + j], s = i / K;
                yg[j] = g + (size_t)i * F; yu[j] = u + (size_t)i * F;
                xp[j] = xq + (size_t)s * H; dp[j] = xdx + (size_t)s * (H / 32); sp[j] = xxs + (size_t)s * (H / 16);
            }
            if (planar)
                for (int o = r0; o < r1; o++) {
                    const uint8_t *wg = ex->g4 + (size_t)o * xf_row_bytes(H), *wu = ex->u4 + (size_t)o * xf_row_bytes(H);
                    const float *sg = ex->gs + (size_t)o * (H / XF_BLOCK), *su = ex->us + (size_t)o * (H / XF_BLOCK);
                    for (int j = 0; j < m; j++) { yg[j][o] = xf_dot_a32(wg, sg, xp[j], dp[j], sp[j], H); yu[j][o] = xf_dot_a32(wu, su, xp[j], dp[j], sp[j], H); }
                }
            else
                for (int o = r0; o < r1; o += 16) {
                    xf16_rows(yg, o, ex->g4 + (size_t)(o / 16) * gbH, xp, dp, sp, H, m);
                    xf16_rows(yu, o, ex->u4 + (size_t)(o / 16) * gbH, xp, dp, sp, H, m);
                }
        }
    }
    /* hidden rows, quantized once per pair */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)n; i++) {
        if (idx[i] < 0 || !experts[i]) continue;
        xf_swiglu(h + (size_t)i * F, g + (size_t)i * F, u + (size_t)i * F, F);
        xf_act32(h + (size_t)i * F, F, hq + (size_t)i * F, hdx + (size_t)i * (F / 32), hxs + (size_t)i * (F / 16));
    }
    /* phase 2: down for every (expert, chunk) */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int it = 0; it < nu * cH; it++) {
        int e = it / cH, c = it % cH;
        int r0 = c * rowsH, r1 = r0 + rowsH; if (r1 > H) r1 = H;
        const XfExpert *ex = experts[ord[start[e]]];
        for (int j0 = start[e]; j0 < start[e + 1]; j0 += XF16_PAIRS) {
            const int m = start[e + 1] - j0 < XF16_PAIRS ? start[e + 1] - j0 : XF16_PAIRS;
            float *yd[XF16_PAIRS]; const int8_t *xp[XF16_PAIRS]; const float *dp[XF16_PAIRS]; const int32_t *sp[XF16_PAIRS];
            for (int j = 0; j < m; j++) {
                const int i = ord[j0 + j];
                yd[j] = ctb + (size_t)i * H; xp[j] = hq + (size_t)i * F; dp[j] = hdx + (size_t)i * (F / 32); sp[j] = hxs + (size_t)i * (F / 16);
            }
            if (planar)
                for (int o = r0; o < r1; o++) {
                    const uint8_t *wd = ex->d4 + (size_t)o * xf_row_bytes(F); const float *sd = ex->ds + (size_t)o * (F / XF_BLOCK);
                    for (int j = 0; j < m; j++) yd[j][o] = xf_dot_a32(wd, sd, xp[j], dp[j], sp[j], F);
                }
            else
                for (int o = r0; o < r1; o += 16) xf16_rows(yd, o, ex->d4 + (size_t)(o / 16) * gbF, xp, dp, sp, F, m);
        }
    }
#undef XF16_PAIRS
    return ctb;
}
#endif

/* The per-(s,k) expert outputs alone: ctb[(s*K+k)*H ..] = expert_k(x[s]) for every
 * pair with an expert, nothing summed. ctb may be the caller's (an engine that adds
 * other outputs in the same rank order, vk_tier.h) or NULL: then it is carved from
 * scratch. Returns it, or NULL when no pair has an expert (nothing written). */
static inline float *xf_moe_compute(float *ctb_out, const float *x, int S, int K, int H, int F,
                                    const int *idx, const XfExpert *const *experts,
                                    int mode, void *scratch) {
#ifdef XF16_ON
    if (mode == XF_MODE_R16 || mode == XF_MODE_A32) return xf_moe_compute_r16(ctb_out, x, S, K, H, F, idx, experts, scratch, mode == XF_MODE_A32);
#endif
    const size_t n = (size_t)S * K;
    char *p = (char *)scratch;
#define XF_TAKE(T, count) ((T *)p); p += (((size_t)(count) * sizeof(T)) + 63) & ~(size_t)63
    float *g    = XF_TAKE(float, n * F);
    float *u    = XF_TAKE(float, n * F);
    float *h    = XF_TAKE(float, n * F);
    float *ctb  = XF_TAKE(float, n * H);
    int8_t *hq  = XF_TAKE(int8_t, n * F);
    float *hsum = XF_TAKE(float, n * (F / XF_BLOCK));
    float *hsx  = XF_TAKE(float, n);
    int8_t *xq  = XF_TAKE(int8_t, (size_t)S * H);
    float *xsum = XF_TAKE(float, (size_t)S * (H / XF_BLOCK));
    float *xsx  = XF_TAKE(float, S);
    int *uniq   = XF_TAKE(int, n);          /* distinct expert ids, in first-seen order */
    int *head   = XF_TAKE(int, n);          /* first (s,k) pair index of each distinct expert */
    int *next   = XF_TAKE(int, n);          /* chain of pairs sharing an expert */
    int *cnt    = XF_TAKE(int, n);
#undef XF_TAKE
    if (ctb_out) ctb = ctb_out;

    /* activations once per token */
    if (mode) for (int s = 0; s < S; s++) xsx[s] = xf_act_i8(x + (size_t)s * H, H, xq + (size_t)s * H, xsum + (size_t)s * (H / XF_BLOCK));

    /* group the (s,k) pairs by expert */
    int nu = 0;
    for (int s = 0; s < S; s++) for (int k = 0; k < K; k++) {
        int i = s * K + k; next[i] = -1;
        if (idx[i] < 0 || !experts[i]) continue;
        int j = 0; for (; j < nu; j++) if (uniq[j] == idx[i]) break;
        if (j == nu) { uniq[nu] = idx[i]; head[nu] = i; cnt[nu] = 1; nu++; }
        else { int t = head[j]; while (next[t] >= 0) t = next[t]; next[t] = i; cnt[j]++; }
    }
    if (nu == 0) return NULL;

    int T = 1;
#ifdef _OPENMP
    T = omp_get_max_threads();
#endif
    /* chunking: enough items to feed every thread, rows in multiples of 8 */
    int cF = (4 * T + nu - 1) / nu; if (cF < 1) cF = 1; if (cF > F / 8) cF = F / 8 > 0 ? F / 8 : 1;
    int cH = (4 * T + nu - 1) / nu; if (cH < 1) cH = 1; if (cH > H / 8) cH = H / 8 > 0 ? H / 8 : 1;
    int rowsF = (F + cF - 1) / cF; rowsF = (rowsF + 7) & ~7; cF = (F + rowsF - 1) / rowsF;
    int rowsH = (H + cH - 1) / cH; rowsH = (rowsH + 7) & ~7; cH = (H + rowsH - 1) / rowsH;

    /* phase 1: gate+up for every (expert, chunk), all rows routed to that expert */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int it = 0; it < nu * cF; it++) {
        int e = it / cF, c = it % cF;
        int r0 = c * rowsF, r1 = r0 + rowsF; if (r1 > F) r1 = F;
        for (int i = head[e]; i >= 0; i = next[i]) {
            int s = i / K;
            XfAct a = { x + (size_t)s * H, xq + (size_t)s * H, xsum + (size_t)s * (H / XF_BLOCK), mode ? xsx[s] : 0.f };
            xf_gate_up_rows(g + (size_t)i * F, u + (size_t)i * F, experts[i], &a, H, r0, r1, mode);
        }
    }
    /* hidden rows (cheap, per pair) */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)n; i++) {
        if (idx[i] < 0 || !experts[i]) continue;
        xf_swiglu(h + (size_t)i * F, g + (size_t)i * F, u + (size_t)i * F, F);
        if (mode) hsx[i] = xf_act_i8(h + (size_t)i * F, F, hq + (size_t)i * F, hsum + (size_t)i * (F / XF_BLOCK));
    }
    /* phase 2: down for every (expert, chunk) */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int it = 0; it < nu * cH; it++) {
        int e = it / cH, c = it % cH;
        int r0 = c * rowsH, r1 = r0 + rowsH; if (r1 > H) r1 = H;
        for (int i = head[e]; i >= 0; i = next[i]) {
            XfAct a = { h + (size_t)i * F, hq + (size_t)i * F, hsum + (size_t)i * (F / XF_BLOCK), mode ? hsx[i] : 0.f };
            xf_down_rows(ctb + (size_t)i * H, experts[i], &a, F, r0, r1, mode);
        }
    }
    return ctb;
}

static inline void xf_moe_add(float *out, const float *x, int S, int K, int H, int F,
                              const int *idx, const float *val, const XfExpert *const *experts,
                              int mode, void *scratch) {
    float *ctb = xf_moe_compute(NULL, x, S, K, H, F, idx, experts, mode, scratch);
    if (!ctb) return;
    /* rank-order sum per token: out[s] += sum_k val[s][k] * ctb[s][k] */
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < S; s++) {
        float *os = out + (size_t)s * H;
        for (int k = 0; k < K; k++) {
            int i = s * K + k; if (idx[i] < 0 || !experts[i]) continue;
            float w = val[i]; const float *c = ctb + (size_t)i * H;
            for (int d = 0; d < H; d++) os[d] += w * c[d];
        }
    }
}

static inline void xf_moe_run(float *out, const float *x, int S, int K, int H, int F,
                              const int *idx, const float *val, const XfExpert *const *experts,
                              int mode, void *scratch) {
    memset(out, 0, (size_t)S * H * sizeof(float));
    xf_moe_add(out, x, S, K, H, F, idx, val, experts, mode, scratch);
}

#endif /* COLI_EXPERT_FFN_H */
