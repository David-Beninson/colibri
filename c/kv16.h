/* kv16.h -- a KV cache in float16: half the RAM of float32. Each value is clamped to
 * float16's range (+-65504: a value past it is the largest one, never an infinity) and
 * rounded to the nearest float16, ties to even, the IEEE rounding: the CPU's (F16C, or the
 * integer path below), a CUDA kernel's (__float2half_rn) and a Vulkan shader's (the same
 * integer arithmetic as kv16_round_bits) give the same bits, and a row read back widens
 * to the float it was rounded to.
 *
 * Float16, not bfloat16: three more bits of mantissa. On Muse Spark's tiny oracle, whose
 * heads' outputs are normalised (a head whose values cancel magnifies their rounding), K
 * and V in bf16 moved the teacher-forced NLL from 2.0790 to 2.2305 nats a token, in f16
 * to 2.0752.
 *
 * kv16_dot_lanes is dot_f32_lanes (matmul_f32.h) over a float16 row: the same lanes in
 * the same order, so a score is the same bits whether its key comes from the cache or
 * from a step's rounded scratch. */
#ifndef COLI_KV16_H
#define COLI_KV16_H
#include <stdint.h>
#include <string.h>
#include "matmul_f32.h"   /* MATMUL_F32_LANES, MATMUL_F32_MADD */

#define KV16_MAX 65504.0f

/* the float32 bits of x rounded to float16's precision (x within +-65504): the integer
 * reference the Vulkan shader (chain_ew.comp, op 11) follows line for line */
static inline uint32_t kv16_round_bits(uint32_t u) {
    uint32_t s = u & 0x80000000u, a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return u | 0x400000u;            /* a NaN stays one */
    if (a >= 0x38800000u)                                  /* float16's normals: 10 bits of mantissa */
        return s | ((a + 0xfffu + ((a >> 13) & 1u)) & 0xffffe000u);
    /* below 2^-14, float16's subnormals: multiples of 2^-24 */
    uint32_t e = a >> 23, m = (a & 0x7fffffu) | 0x800000u;
    if (e < 102) return s;                                 /* under 2^-25: zero (2^-25 itself a tie, to even) */
    uint32_t sh = 126 - e;                                 /* 14 .. 24: the value is m * 2^-(sh + 24) */
    uint32_t n = (m + (1u << (sh - 1)) - 1u + ((m >> sh) & 1u)) >> sh;
    float f = (float)n * 5.9604644775390625e-8f;           /* n * 2^-24, exact */
    uint32_t r; memcpy(&r, &f, 4);
    return s | r;
}
static inline float kv16_clamp(float x) { return x > KV16_MAX ? KV16_MAX : x < -KV16_MAX ? -KV16_MAX : x; }

#if defined(__FLT16_MAX__) && !defined(KV16_INTEGER)   /* KV16_INTEGER: the integer path (tests) */
typedef _Float16 kv16_t;
static inline uint16_t kv16_from(float f) { kv16_t h = (kv16_t)kv16_clamp(f); uint16_t b; memcpy(&b, &h, 2); return b; }
static inline float kv16_to(uint16_t b) { kv16_t h; memcpy(&h, &b, 2); return (float)h; }
#else
/* no _Float16 in this compiler: the bits by integer arithmetic */
static inline uint16_t kv16_from(float f) {
    f = kv16_clamp(f);
    uint32_t u; memcpy(&u, &f, 4);
    u = kv16_round_bits(u);
    uint32_t s = (u >> 16) & 0x8000u, a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return (uint16_t)(s | 0x7e00u);
    if (a < 0x38800000u) { float g; memcpy(&g, &a, 4); return (uint16_t)(s | (uint32_t)(g * 16777216.0f)); }   /* n * 2^-24 */
    return (uint16_t)(s | (((a >> 23) - 112u) << 10) | ((a >> 13) & 0x3ffu));
}
static inline float kv16_to(uint16_t b) {
    uint32_t s = ((uint32_t)b & 0x8000u) << 16, e = (b >> 10) & 0x1fu, m = b & 0x3ffu, u;
    if (e == 0) { float g = (float)m * 5.9604644775390625e-8f; memcpy(&u, &g, 4); u |= s; }
    else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112u) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4);
    return f;
}
#endif

#if defined(__ARM_NEON) && defined(__FLT16_MAX__) && !defined(KV16_INTEGER)
#include <arm_neon.h>
#define KV16_NEON 1
static inline float32x4_t kv16_clamp4(float32x4_t v) {
    /* match kv16_clamp: x > MAX ? MAX : x < -MAX ? -MAX : x (NaN passes through) */
    uint32x4_t gt = vcgtq_f32(v, vdupq_n_f32(KV16_MAX));
    uint32x4_t lt = vcltq_f32(v, vdupq_n_f32(-KV16_MAX));
    v = vbslq_f32(lt, vdupq_n_f32(-KV16_MAX), v);
    return vbslq_f32(gt, vdupq_n_f32(KV16_MAX), v);
}
#endif
/* x rounded in place to the floats its float16 holds */
static inline void kv16_round(float *x, int64_t n) {
    int64_t i = 0;
#ifdef KV16_NEON
    for (; i + 8 <= n; i += 8) {
        float16x4_t h0 = vcvt_f16_f32(kv16_clamp4(vld1q_f32(x + i)));
        float16x4_t h1 = vcvt_f16_f32(kv16_clamp4(vld1q_f32(x + i + 4)));
        vst1q_f32(x + i,     vcvt_f32_f16(h0));
        vst1q_f32(x + i + 4, vcvt_f32_f16(h1));
    }
#endif
    for (; i < n; i++) x[i] = kv16_to(kv16_from(x[i]));
}
static inline void kv16_pack(uint16_t *d, const float *s, int64_t n) {
    int64_t i = 0;
#ifdef KV16_NEON
    for (; i + 8 <= n; i += 8) {
        float16x4_t h0 = vcvt_f16_f32(kv16_clamp4(vld1q_f32(s + i)));
        float16x4_t h1 = vcvt_f16_f32(kv16_clamp4(vld1q_f32(s + i + 4)));
        vst1q_f16((__fp16 *)(d + i), vcombine_f16(h0, h1));
    }
#endif
    for (; i < n; i++) d[i] = kv16_from(s[i]);
}
static inline void kv16_unpack(float *d, const uint16_t *s, int64_t n) {
    int64_t i = 0;
#ifdef KV16_NEON
    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16 *)(s + i));
        vst1q_f32(d + i,     vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(d + i + 4, vcvt_f32_f16(vget_high_f16(h)));
    }
#endif
    for (; i < n; i++) d[i] = kv16_to(s[i]);
}

static inline float kv16_dot_lanes(const float *x, const uint16_t *w, int I) {
    float a[MATMUL_F32_LANES] = {0};
    int i = 0;
#if defined(KV16_NEON) && MATMUL_F32_LANES == 32
    float32x4_t c0 = vdupq_n_f32(0.f), c1 = c0, c2 = c0, c3 = c0;
    float32x4_t c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    for (; i + 32 <= I; i += 32) {
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
        for (int j = 0; j < MATMUL_F32_LANES; j++) a[j] = MATMUL_F32_MADD(a[j], x[i+j], kv16_to(w[i+j]));
#endif
    for (int j = 0; i + j < I; j++) a[j] = MATMUL_F32_MADD(a[j], x[i+j], kv16_to(w[i+j]));
    for (int h = MATMUL_F32_LANES/2; h > 0; h >>= 1)
        for (int j = 0; j < h; j++) a[j] += a[j+h];
    return a[0];
}
/* y += a * x, the float16 row widened (the float row's own loop: y + a*x, fused). */
static inline void kv16_axpy(float *y, float a, const uint16_t *x, int n) {
    int i = 0;
#ifdef KV16_NEON
    float32x4_t va = vdupq_n_f32(a);
    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16 *)(x + i));
        vst1q_f32(y + i,     vfmaq_f32(vld1q_f32(y + i),     va, vcvt_f32_f16(vget_low_f16(h))));
        vst1q_f32(y + i + 4, vfmaq_f32(vld1q_f32(y + i + 4), va, vcvt_f32_f16(vget_high_f16(h))));
    }
#endif
    for (; i < n; i++) y[i] = MATMUL_F32_MADD(y[i], a, kv16_to(x[i]));
}
#endif
