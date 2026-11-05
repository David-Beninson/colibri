#ifndef COLIBRI_QGEMV_H
#define COLIBRI_QGEMV_H
/* Plain (non-group-scaled) int8 GEMV: one f32 scale per output row, q[O,I]
 * int8 row-major. Its own header so tests/test_qgemv.c can link the very
 * kernel the engine runs; qwen36.c carries a main and cannot be linked into
 * a test.
 *
 * This is qwen36's dense-projection kernel (lm_head and any non-expert
 * quantized matmul); the group-scaled sibling used for expert matmuls lives
 * in gsgemv.h. Its float operations must stay in exactly this order -- float
 * addition is not associative and the engine's token stream is required to
 * be byte-identical to the reference. tests/test_qgemv.c holds the
 * pre-restructure kernel verbatim and compares raw float bits, so any
 * reassociation fails there rather than surfacing as drifted text much
 * later.
 *
 * The SSE4.1 tier below is the exception to that byte-identical rule, by
 * design: it is new code with no pre-existing output to match, and it is
 * checked by tests/test_qgemv.c against a tolerance, not memcmp. It routes
 * its FMA and float loads through sse41_kernels.h -- the same shared
 * primitives header olmoe.c uses -- rather than inlining its own copy. */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__ARM_NEON)
static inline int32_t dot_i8_16(const int8_t *a, const int8_t *b) {
    int32x4_t acc = vdupq_n_s32(0);
    int8x16_t va = vld1q_s8(a), vb = vld1q_s8(b);
#if defined(__ARM_FEATURE_DOTPROD)
    acc = vdotq_s32(acc, va, vb);
#else
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va),  vget_low_s8(vb)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
#endif
    return vaddvq_s32(acc);
}
#endif

static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
#if defined(__ARM_NEON)
    /* IDOT is opt-in, not default-on: this path quantizes the ACTIVATIONS to
     * Q8_0 per 16-element block, which the scalar path does not, so the two are
     * not numerically equivalent. olmoe shipped it default-on and it cost
     * token-exactness end to end (#1044, fixed in af48fe8 by making it opt-in);
     * qwen36 inherited the same default from the same family of kernels. The
     * tiny-oracle gate would not have caught it -- that job runs on x86. */
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = (e && atoi(e)); }
    if (idot && I % 16 == 0 && I <= 4096) {
        int nb = I / 16; int8_t xi[4096]; float xs[256];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b*16;
            float am = 0.f; for (int i = 0; i < 16; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
            float s = am/127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f/s;
            for (int i = 0; i < 16; i++) xi[b*16+i] = (int8_t)lrintf(xb[i]*inv);
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            float acc = 0.f;
            for (int b = 0; b < nb; b++) acc += xs[b]*(float)dot_i8_16(xi+b*16, w+b*16);
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
#if defined(__ARM_NEON)
    #pragma omp parallel for schedule(static) if(O >= 256)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
        float32x4_t a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
        int i = 0;
        for (; i + 16 <= I; i += 16) {
            int8x16_t b0 = vld1q_s8(w + i);
            int16x8_t wlo = vmovl_s8(vget_low_s8(b0));
            int16x8_t whi = vmovl_s8(vget_high_s8(b0));
            a0 = vfmaq_f32(a0, vld1q_f32(x + i),      vcvtq_f32_s32(vmovl_s16(vget_low_s16(wlo))));
            a1 = vfmaq_f32(a1, vld1q_f32(x + i + 4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(wlo))));
            a2 = vfmaq_f32(a2, vld1q_f32(x + i + 8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(whi))));
            a3 = vfmaq_f32(a3, vld1q_f32(x + i + 12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(whi))));
        }
        float32x4_t s = vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3));
        float acc = vaddvq_f32(s);
        for (; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
#else
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
#endif
}

#endif /* COLIBRI_QGEMV_H */
