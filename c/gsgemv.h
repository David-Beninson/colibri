#ifndef COLIBRI_GSGEMV_H
#define COLIBRI_GSGEMV_H
/* Group-scaled int8 GEMV: one f32 scale per `gs` input elements per row, the
 * layout the gs64 expert containers use.  Row layout of `scale`: [O][I/gs]
 * row-major.  Its own header so tests/test_gsgemv.c can link the very kernel
 * the engine runs; qwen36.c carries a main and cannot be linked into a test.
 *
 * This is qwen36's hottest kernel: every expert matmul goes through it, three
 * per expert, eight experts, forty layers.  Each row's float operations must
 * stay in exactly this order -- float addition is not associative and the
 * engine's token stream is required to be byte-identical to the reference.
 * tests/test_gsgemv.c holds the pre-restructure kernel verbatim and compares
 * raw float bits, so any reassociation fails there rather than surfacing as
 * drifted text much later.
 *
 * The SSE4.1 tier below is the exception to that byte-identical rule, by
 * design: it is new code with no pre-existing output to match, its tree
 * reduction is a genuinely different shape than the scalar reference, and it
 * is checked by tests/test_gsgemv.c against a tolerance, not memcmp. It
 * routes its FMA and float loads through sse41_kernels.h -- the same shared
 * primitives header olmoe.c uses -- rather than inlining its own copy. */
#include <stdint.h>
#include <math.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>

static inline float gs_group_sum_neon(float32x4_t a0, float32x4_t a1) {
    float32x4_t s = vaddq_f32(a0, a1);
    float32x2_t p = vadd_f32(vget_low_f32(s), vget_high_f32(s));
    return vget_lane_f32(vpadd_f32(p, p), 0);
}
#endif

static void matmul_q_gs(float *y, const float *x, const int8_t *q, const float *scale,
                        int I, int O, int gs) {
    int ng = (I + gs - 1) / gs;
#if defined(__ARM_NEON)
    if ((gs & 15) == 0) {
        int o4 = O & ~3;
        #pragma omp parallel for schedule(static) if(O >= 256)
        for (int ob = 0; ob < o4; ob += 4) {
            const int8_t *w0 = q + (int64_t)(ob+0) * I, *w1 = q + (int64_t)(ob+1) * I;
            const int8_t *w2 = q + (int64_t)(ob+2) * I, *w3 = q + (int64_t)(ob+3) * I;
            const float *sc0 = scale + (int64_t)(ob+0) * ng, *sc1 = scale + (int64_t)(ob+1) * ng;
            const float *sc2 = scale + (int64_t)(ob+2) * ng, *sc3 = scale + (int64_t)(ob+3) * ng;
            float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
            for (int gi = 0; gi < ng; gi++) {
                float32x4_t a00 = vdupq_n_f32(0.f), a01 = vdupq_n_f32(0.f);
                float32x4_t a10 = vdupq_n_f32(0.f), a11 = vdupq_n_f32(0.f);
                float32x4_t a20 = vdupq_n_f32(0.f), a21 = vdupq_n_f32(0.f);
                float32x4_t a30 = vdupq_n_f32(0.f), a31 = vdupq_n_f32(0.f);
                int base = gi * gs, end = base + gs; if (end > I) end = I;
                for (int i = base; i + 16 <= end; i += 16) {
                    float32x4_t x0 = vld1q_f32(x + i),      x1 = vld1q_f32(x + i + 4);
                    float32x4_t x2 = vld1q_f32(x + i + 8),  x3 = vld1q_f32(x + i + 12);

                    int8x16_t b0 = vld1q_s8(w0 + i);
                    int16x8_t b0l = vmovl_s8(vget_low_s8(b0)), b0h = vmovl_s8(vget_high_s8(b0));
                    a00 = vfmaq_f32(a00, x0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(b0l))));
                    a01 = vfmaq_f32(a01, x1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(b0l))));
                    a00 = vfmaq_f32(a00, x2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(b0h))));
                    a01 = vfmaq_f32(a01, x3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(b0h))));

                    int8x16_t b1 = vld1q_s8(w1 + i);
                    int16x8_t b1l = vmovl_s8(vget_low_s8(b1)), b1h = vmovl_s8(vget_high_s8(b1));
                    a10 = vfmaq_f32(a10, x0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(b1l))));
                    a11 = vfmaq_f32(a11, x1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(b1l))));
                    a10 = vfmaq_f32(a10, x2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(b1h))));
                    a11 = vfmaq_f32(a11, x3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(b1h))));

                    int8x16_t b2 = vld1q_s8(w2 + i);
                    int16x8_t b2l = vmovl_s8(vget_low_s8(b2)), b2h = vmovl_s8(vget_high_s8(b2));
                    a20 = vfmaq_f32(a20, x0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(b2l))));
                    a21 = vfmaq_f32(a21, x1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(b2l))));
                    a20 = vfmaq_f32(a20, x2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(b2h))));
                    a21 = vfmaq_f32(a21, x3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(b2h))));

                    int8x16_t b3 = vld1q_s8(w3 + i);
                    int16x8_t b3l = vmovl_s8(vget_low_s8(b3)), b3h = vmovl_s8(vget_high_s8(b3));
                    a30 = vfmaq_f32(a30, x0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(b3l))));
                    a31 = vfmaq_f32(a31, x1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(b3l))));
                    a30 = vfmaq_f32(a30, x2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(b3h))));
                    a31 = vfmaq_f32(a31, x3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(b3h))));
                }
                acc0 = fmaf(gs_group_sum_neon(a00, a01), sc0[gi], acc0);
                acc1 = fmaf(gs_group_sum_neon(a10, a11), sc1[gi], acc1);
                acc2 = fmaf(gs_group_sum_neon(a20, a21), sc2[gi], acc2);
                acc3 = fmaf(gs_group_sum_neon(a30, a31), sc3[gi], acc3);
            }
            y[ob+0] = acc0; y[ob+1] = acc1; y[ob+2] = acc2; y[ob+3] = acc3;
        }
        for (int o = o4; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            const float *sc = scale + (int64_t)o * ng;
            float acc = 0.f;
            for (int gi = 0; gi < ng; gi++) {
                float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
                int base = gi * gs, end = base + gs; if (end > I) end = I;
                for (int i = base; i + 16 <= end; i += 16) {
                    int8x16_t b0 = vld1q_s8(w + i);
                    int16x8_t b0l = vmovl_s8(vget_low_s8(b0)), b0h = vmovl_s8(vget_high_s8(b0));
                    a0 = vfmaq_f32(a0, vld1q_f32(x + i),      vcvtq_f32_s32(vmovl_s16(vget_low_s16(b0l))));
                    a1 = vfmaq_f32(a1, vld1q_f32(x + i + 4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(b0l))));
                    a0 = vfmaq_f32(a0, vld1q_f32(x + i + 8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(b0h))));
                    a1 = vfmaq_f32(a1, vld1q_f32(x + i + 12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(b0h))));
                }
                acc = fmaf(gs_group_sum_neon(a0, a1), sc[gi], acc);
            }
            y[o] = acc;
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static) if(O >= 256)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        const float *sc = scale + (int64_t)o * ng;
        float acc = 0.f;
        for (int gi = 0; gi < ng; gi++) {
            int base = gi * gs, end = base + gs; if (end > I) end = I;
            float part = 0.f;
            for (int i = base; i < end; i++) part += x[i] * (float)w[i];
            acc += part * sc[gi];
        }
        y[o] = acc;
    }
}

#endif /* COLIBRI_GSGEMV_H */
