/* matmul_f16.h — y[S,O] = x[S,I] @ W^T with W[O,I] stored as IEEE half or BF16 (header-only, static).
 *
 * Same contract as matmul_f32.h: the weight is widened to f32 (exact) and every dot product
 * runs the very same fixed lanes, the same fused multiply-add and the same halving tree as
 * dot_f32_lanes(). The result is therefore bit-identical to matmul() on the widened matrix
 * (tests/test_matmul_f16.c compares the bits), while the matrix costs half the DRAM bytes.
 * A decode step that streams its attention and lm_head weights from memory is
 * bandwidth-bound, so keeping a half checkpoint's tensors half-sized is a straight win.
 *
 * IEEE half needs F16C to widen 8 halves per instruction; without it matmul_f16w is not
 * defined (COLI_MATMUL_F16 stays undefined). BF16 needs nothing: its widening is a 16-bit
 * shift, so matmul_bf16w is defined on every target, ARM included. */
#ifndef COLI_MATMUL_F16_H
#define COLI_MATMUL_F16_H

#include <string.h>
#include "matmul_f32.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define COLI_MATMUL_F16 1

static inline float f16w_to_f32_scalar(uint16_t b){
    __fp16 h; memcpy(&h, &b, 2); return (float)h;
}

static inline float dot_f16w_lanes(const float *x, const uint16_t *w, int I){
    float a[MATMUL_F32_LANES] = {0};
    int i = 0;
#if MATMUL_F32_LANES == 32
    float32x4_t c0 = vdupq_n_f32(0.f), c1 = c0, c2 = c0, c3 = c0;
    float32x4_t c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    for (; i + 32 <= I; i += 32){
        float16x8_t h0 = vld1q_f16((const __fp16 *)(w + i));
        float16x8_t h1 = vld1q_f16((const __fp16 *)(w + i + 8));
        float16x8_t h2 = vld1q_f16((const __fp16 *)(w + i + 16));
        float16x8_t h3 = vld1q_f16((const __fp16 *)(w + i + 24));
        c0 = vfmaq_f32(c0, vld1q_f32(x + i),      vcvt_f32_f16(vget_low_f16(h0)));
        c1 = vfmaq_f32(c1, vld1q_f32(x + i + 4),  vcvt_f32_f16(vget_high_f16(h0)));
        c2 = vfmaq_f32(c2, vld1q_f32(x + i + 8),  vcvt_f32_f16(vget_low_f16(h1)));
        c3 = vfmaq_f32(c3, vld1q_f32(x + i + 12), vcvt_f32_f16(vget_high_f16(h1)));
        c4 = vfmaq_f32(c4, vld1q_f32(x + i + 16), vcvt_f32_f16(vget_low_f16(h2)));
        c5 = vfmaq_f32(c5, vld1q_f32(x + i + 20), vcvt_f32_f16(vget_high_f16(h2)));
        c6 = vfmaq_f32(c6, vld1q_f32(x + i + 24), vcvt_f32_f16(vget_low_f16(h3)));
        c7 = vfmaq_f32(c7, vld1q_f32(x + i + 28), vcvt_f32_f16(vget_high_f16(h3)));
    }
    vst1q_f32(a,      c0); vst1q_f32(a + 4,  c1); vst1q_f32(a + 8,  c2); vst1q_f32(a + 12, c3);
    vst1q_f32(a + 16, c4); vst1q_f32(a + 20, c5); vst1q_f32(a + 24, c6); vst1q_f32(a + 28, c7);
#else
    for (; i + MATMUL_F32_LANES <= I; i += MATMUL_F32_LANES)
        for (int j = 0; j < MATMUL_F32_LANES; j++) a[j] = MATMUL_F32_MADD(a[j], x[i+j], f16w_to_f32_scalar(w[i+j]));
#endif
    for (int j = 0; i + j < I; j++) a[j] = MATMUL_F32_MADD(a[j], x[i+j], f16w_to_f32_scalar(w[i+j]));
    for (int h = MATMUL_F32_LANES/2; h > 0; h >>= 1)
        for (int j = 0; j < h; j++) a[j] += a[j+h];
    return a[0];
}

static void matmul_f16w(float *y, const float *x, const uint16_t *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++){
        const uint16_t *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) y[(int64_t)s * O + o] = dot_f16w_lanes(x + (int64_t)s * I, w, I);
    }
}
#endif

static inline float bf16w_to_f32(uint16_t h){
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, sizeof f); return f;
}

static inline float dot_bf16w_lanes(const float *x, const uint16_t *w, int I){
    float a[MATMUL_F32_LANES] = {0};
    int i = 0;
#if defined(__ARM_NEON) && MATMUL_F32_LANES == 32
    float32x4_t c0 = vdupq_n_f32(0.f), c1 = c0, c2 = c0, c3 = c0;
    float32x4_t c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    for (; i + 32 <= I; i += 32){
        uint16x8_t b0 = vld1q_u16(w + i),      b1 = vld1q_u16(w + i + 8);
        uint16x8_t b2 = vld1q_u16(w + i + 16), b3 = vld1q_u16(w + i + 24);
        c0 = vfmaq_f32(c0, vld1q_f32(x + i),      vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b0), 16)));
        c1 = vfmaq_f32(c1, vld1q_f32(x + i + 4),  vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b0), 16)));
        c2 = vfmaq_f32(c2, vld1q_f32(x + i + 8),  vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b1), 16)));
        c3 = vfmaq_f32(c3, vld1q_f32(x + i + 12), vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b1), 16)));
        c4 = vfmaq_f32(c4, vld1q_f32(x + i + 16), vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b2), 16)));
        c5 = vfmaq_f32(c5, vld1q_f32(x + i + 20), vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b2), 16)));
        c6 = vfmaq_f32(c6, vld1q_f32(x + i + 24), vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b3), 16)));
        c7 = vfmaq_f32(c7, vld1q_f32(x + i + 28), vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b3), 16)));
    }
    vst1q_f32(a,      c0); vst1q_f32(a + 4,  c1); vst1q_f32(a + 8,  c2); vst1q_f32(a + 12, c3);
    vst1q_f32(a + 16, c4); vst1q_f32(a + 20, c5); vst1q_f32(a + 24, c6); vst1q_f32(a + 28, c7);
#else
    for (; i + MATMUL_F32_LANES <= I; i += MATMUL_F32_LANES)
        for (int j = 0; j < MATMUL_F32_LANES; j++) a[j] = MATMUL_F32_MADD(a[j], x[i+j], bf16w_to_f32(w[i+j]));
#endif
    for (int j = 0; i + j < I; j++) a[j] = MATMUL_F32_MADD(a[j], x[i+j], bf16w_to_f32(w[i+j]));
    for (int h = MATMUL_F32_LANES/2; h > 0; h >>= 1)
        for (int j = 0; j < h; j++) a[j] += a[j+h];
    return a[0];
}

static void matmul_bf16w(float *y, const float *x, const uint16_t *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++){
        const uint16_t *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) y[(int64_t)s * O + o] = dot_bf16w_lanes(x + (int64_t)s * I, w, I);
    }
}

#endif
