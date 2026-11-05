/* idot.h -- integer dot-product kernels shared by every engine.
 *
 * The activation is quantized to int8 once (one scale per row, amax/127,
 * qrow_i8) and the weights, int8 rows or int4 planar blocks with per-row or
 * per-group scales, are multiplied with integer instructions: maddubs on SSSE3 and
 * AVX2, vpdpbusd on AVX-VNNI and AVX-512 VNNI, sdot/smmla on NEON, and the
 * AMX tile kernel where the CPU has it. Exact int32 sums, scaled once per
 * output, so every ISA path is bit-identical to the scalar reference.
 *
 * Lived inside quant.h; moved here so an engine that has its own dense
 * kernels (qwen36.c uses qgemv.h/gsgemv.h, whose matmul_q would collide
 * with quant.h's) can still take the integer path for its dense trunk.
 * quant.h includes this file at the top, so its other consumers see the
 * same declarations in the same place as before. */
#ifndef COLIBRI_IDOT_H
#define COLIBRI_IDOT_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
/* ---- SIMD includes -------------------------------------------------------- */
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ---- IDOT: integer dot kernels (int8-quantized activations) --------------- */
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
#define IDOT_KERNEL "neon-i8mm"
#elif defined(__ARM_NEON)
#define IDOT_KERNEL "neon"
#else
#define IDOT_KERNEL "scalar"
#endif

static inline float qrow_i8(const float *x, int8_t *q, int I){
    float amax=0; for(int i=0;i<I;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    float s=amax/127.f; if(s<1e-12f) s=1e-12f; float inv=1.f/s;
    for(int i=0;i<I;i++) q[i]=(int8_t)lrintf(x[i]*inv);
    return s;
}

/* Activation -> int8 with one scale (shared by every engine whose trunk
 * meets the integer kernels below), plus the int32 sum of every block of 64
 * (the K1b kernel subtracts 8*sum per group because its nibbles are unsigned).
 * Rounding is to nearest even in both paths, so the vector path equals qrow_i8 bit for bit. */
static float dense_act_i8(const float *x, int I, int8_t *xq, int32_t *xsg){
    float amax = 0.f;
    int i = 0;
#if defined(__ARM_NEON)
    {
        float32x4_t am0 = vdupq_n_f32(0.f), am1 = vdupq_n_f32(0.f);
        for (; i + 8 <= I; i += 8) {
            am0 = vmaxq_f32(am0, vabsq_f32(vld1q_f32(x + i)));
            am1 = vmaxq_f32(am1, vabsq_f32(vld1q_f32(x + i + 4)));
        }
        amax = vmaxvq_f32(vmaxq_f32(am0, am1));
    }
#endif
    for (; i < I; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    float s = amax / 127.f; if (s < 1e-12f) s = 1e-12f;
    float inv = 1.f / s;
    i = 0;
#if defined(__ARM_NEON)
    {
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
    }
#endif
    for (; i < I; i++) xq[i] = (int8_t)lrintf(x[i] * inv);
    if (xsg) {
        int ng = I / 64;
        for (int g = 0; g < ng; g++) {
#if defined(__ARM_NEON)
            const int8_t *p = xq + g * 64;
            int32_t sum = (int32_t)vaddlvq_s8(vld1q_s8(p))
                        + (int32_t)vaddlvq_s8(vld1q_s8(p + 16))
                        + (int32_t)vaddlvq_s8(vld1q_s8(p + 32))
                        + (int32_t)vaddlvq_s8(vld1q_s8(p + 48));
            xsg[g] = sum;
#else
            int32_t sum = 0;
            for (int k = 0; k < 64; k++) sum += xq[g * 64 + k];
            xsg[g] = sum;
#endif
        }
    }
    return s;
}

/* dot int8*int8 */
static inline int32_t dot_i8i8(const int8_t *w, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#if defined(__ARM_NEON)
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0),a2=vdupq_n_s32(0),a3=vdupq_n_s32(0);
    for(;i+64<=I;i+=64){
        a0=vdotq_s32(a0,vld1q_s8(w+i),   vld1q_s8(x+i));
        a1=vdotq_s32(a1,vld1q_s8(w+i+16),vld1q_s8(x+i+16));
        a2=vdotq_s32(a2,vld1q_s8(w+i+32),vld1q_s8(x+i+32));
        a3=vdotq_s32(a3,vld1q_s8(w+i+48),vld1q_s8(x+i+48));
    }
    int32x4_t acc=vaddq_s32(vaddq_s32(a0,a1),vaddq_s32(a2,a3));
    for(;i+16<=I;i+=16) acc=vdotq_s32(acc,vld1q_s8(w+i),vld1q_s8(x+i));
    sum=vaddvq_s32(acc);
#else
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+16<=I;i+=16){
        int8x16_t wv=vld1q_s8(w+i), xv=vld1q_s8(x+i);
        int16x8_t p=vmull_s8(vget_low_s8(wv),vget_low_s8(xv));
        p=vmlal_s8(p,vget_high_s8(wv),vget_high_s8(xv));
        acc=vpadalq_s16(acc,p);
    }
    sum=vaddvq_s32(acc);
#endif
#endif
    for(;i<I;i++) sum+=(int32_t)w[i]*x[i];
    return sum;
}

/* n (1..4) activation rows against one weight row: out[r] = dot_i8i8(w, x + r*I, I). */
#define IDOT_MAX_ROWS 4
static inline void dot_i8i8_rows(const int8_t *w, const int8_t *x, int I, int n, int32_t *out){
    int i=0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc[IDOT_MAX_ROWS];
    for(int r=0;r<n;r++) acc[r]=vdupq_n_s32(0);
    for(;i+16<=I;i+=16){
        int8x16_t wv=vld1q_s8(w+i);
        for(int r=0;r<n;r++){
            int8x16_t xv=vld1q_s8(x+(size_t)r*I+i);
            acc[r]=vdotq_s32(acc[r],wv,xv);
        }
    }
    for(int r=0;r<n;r++) out[r]=vaddvq_s32(acc[r]);
#else
    for(int r=0;r<n;r++) out[r]=dot_i8i8(w,x+(size_t)r*I,I);
    i=I;
#endif
    for(int r=0;r<n;r++){ int32_t t=0; const int8_t *xr=x+(size_t)r*I; for(int j=i;j<I;j++) t+=(int32_t)w[j]*xr[j]; out[r]+=t; }
}

/* dot int4(packed)*int8 */
static inline int32_t dot_i4i8(const uint8_t *w4, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#if defined(__ARM_NEON)
    const uint8x16_t m4q=vdupq_n_u8(0x0F); const int8x16_t b8q=vdupq_n_s8(8);
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0),a2=vdupq_n_s32(0),a3=vdupq_n_s32(0);
    for(;i+64<=I;i+=64){
        uint8x16_t byA=vld1q_u8(w4+(i>>1)), byB=vld1q_u8(w4+(i>>1)+16);
        uint8x16x2_t zA=vzipq_u8(vandq_u8(byA,m4q), vshrq_n_u8(byA,4));
        uint8x16x2_t zB=vzipq_u8(vandq_u8(byB,m4q), vshrq_n_u8(byB,4));
        a0=vdotq_s32(a0,vsubq_s8(vreinterpretq_s8_u8(zA.val[0]),b8q),vld1q_s8(x+i));
        a1=vdotq_s32(a1,vsubq_s8(vreinterpretq_s8_u8(zA.val[1]),b8q),vld1q_s8(x+i+16));
        a2=vdotq_s32(a2,vsubq_s8(vreinterpretq_s8_u8(zB.val[0]),b8q),vld1q_s8(x+i+32));
        a3=vdotq_s32(a3,vsubq_s8(vreinterpretq_s8_u8(zB.val[1]),b8q),vld1q_s8(x+i+48));
    }
    int32x4_t acc=vaddq_s32(vaddq_s32(a0,a1),vaddq_s32(a2,a3));
    for(;i+32<=I;i+=32){
        uint8x16_t by=vld1q_u8(w4+(i>>1));
        uint8x16x2_t z=vzipq_u8(vandq_u8(by,m4q), vshrq_n_u8(by,4));
        acc=vdotq_s32(acc,vsubq_s8(vreinterpretq_s8_u8(z.val[0]),b8q),vld1q_s8(x+i));
        acc=vdotq_s32(acc,vsubq_s8(vreinterpretq_s8_u8(z.val[1]),b8q),vld1q_s8(x+i+16));
    }
    sum=vaddvq_s32(acc);
#else
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+32<=I;i+=32){
        uint8x16_t by=vld1q_u8(w4+(i>>1));
        uint8x16x2_t z=vzipq_u8(vandq_u8(by,m4q), vshrq_n_u8(by,4));
        int8x16_t w0=vsubq_s8(vreinterpretq_s8_u8(z.val[0]),b8q);
        int8x16_t w1=vsubq_s8(vreinterpretq_s8_u8(z.val[1]),b8q);
        int8x16_t x0=vld1q_s8(x+i), x1=vld1q_s8(x+i+16);
        int16x8_t p=vmull_s8(vget_low_s8(w0),vget_low_s8(x0));
        p=vmlal_s8(p,vget_high_s8(w0),vget_high_s8(x0));
        acc=vpadalq_s16(acc,p);
        p=vmull_s8(vget_low_s8(w1),vget_low_s8(x1));
        p=vmlal_s8(p,vget_high_s8(w1),vget_high_s8(x1));
        acc=vpadalq_s16(acc,p);
    }
    sum=vaddvq_s32(acc);
#endif
#endif
    for(;i+1<I;i+=2){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
    if(i<I){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]; }
    return sum;
}

/* ---- ARM i8mm SMMLA tiled kernels ---------------------------------------- */
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
static inline int32x4_t mm_tile16(int32x4_t acc, int8x16_t wo, int8x16_t wo1,
                                  int8x16_t xs, int8x16_t xs1){
    acc=vmmlaq_s32(acc, vcombine_s8(vget_low_s8(wo), vget_low_s8(wo1)),
                        vcombine_s8(vget_low_s8(xs), vget_low_s8(xs1)));
    return vmmlaq_s32(acc, vcombine_s8(vget_high_s8(wo), vget_high_s8(wo1)),
                           vcombine_s8(vget_high_s8(xs), vget_high_s8(xs1)));
}
static void matmul_q_idot_mm(float *y, const int8_t *xq, const float *sx, const int8_t *q,
                             const float *scale, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<(O&~1);o+=2){
        const int8_t *wo=q+(int64_t)o*I, *wo1=q+(int64_t)(o+1)*I;
        float sc0=scale[o], sc1=scale[o+1];
        for(int s=0;s<(S&~1);s+=2){
            const int8_t *xs=xq+(int64_t)s*I, *xs1=xq+(int64_t)(s+1)*I;
            int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0),a2=vdupq_n_s32(0),a3=vdupq_n_s32(0); int i=0;
            for(;i+64<=I;i+=64){
                a0=mm_tile16(a0,vld1q_s8(wo+i),   vld1q_s8(wo1+i),   vld1q_s8(xs+i),   vld1q_s8(xs1+i));
                a1=mm_tile16(a1,vld1q_s8(wo+i+16),vld1q_s8(wo1+i+16),vld1q_s8(xs+i+16),vld1q_s8(xs1+i+16));
                a2=mm_tile16(a2,vld1q_s8(wo+i+32),vld1q_s8(wo1+i+32),vld1q_s8(xs+i+32),vld1q_s8(xs1+i+32));
                a3=mm_tile16(a3,vld1q_s8(wo+i+48),vld1q_s8(wo1+i+48),vld1q_s8(xs+i+48),vld1q_s8(xs1+i+48));
            }
            for(;i+16<=I;i+=16)
                a0=mm_tile16(a0,vld1q_s8(wo+i),vld1q_s8(wo1+i),vld1q_s8(xs+i),vld1q_s8(xs1+i));
            int32x4_t acc=vaddq_s32(vaddq_s32(a0,a1),vaddq_s32(a2,a3));
            int32_t d00=vgetq_lane_s32(acc,0), d01=vgetq_lane_s32(acc,1);
            int32_t d10=vgetq_lane_s32(acc,2), d11=vgetq_lane_s32(acc,3);
            for(;i<I;i++){ int a=wo[i],b=wo1[i],u=xs[i],v=xs1[i];
                d00+=a*u; d01+=a*v; d10+=b*u; d11+=b*v; }
            y[(int64_t)s*O+o]        =(float)d00*sc0*sx[s];
            y[(int64_t)s*O+(o+1)]    =(float)d10*sc1*sx[s];
            y[(int64_t)(s+1)*O+o]    =(float)d01*sc0*sx[s+1];
            y[(int64_t)(s+1)*O+(o+1)]=(float)d11*sc1*sx[s+1];
        }
        if(S&1){ int s=S-1; const int8_t *xs=xq+(int64_t)s*I;
            y[(int64_t)s*O+o]    =(float)dot_i8i8(wo, xs,I)*sc0*sx[s];
            y[(int64_t)s*O+(o+1)]=(float)dot_i8i8(wo1,xs,I)*sc1*sx[s]; }
    }
    if(O&1){ int o=O-1; const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        #pragma omp parallel for schedule(static)
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i8i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}
static void matmul_i4_idot_mm(float *y, const int8_t *xq, const float *sx, const uint8_t *q4,
                              const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<(O&~1);o+=2){
        const uint8x16_t m4q=vdupq_n_u8(0x0F); const int8x16_t b8q=vdupq_n_s8(8);
        const uint8_t *wo=q4+(int64_t)o*rb, *wo1=q4+(int64_t)(o+1)*rb;
        float sc0=scale[o], sc1=scale[o+1];
        for(int s=0;s<(S&~1);s+=2){
            const int8_t *xs=xq+(int64_t)s*I, *xs1=xq+(int64_t)(s+1)*I;
            int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0),a2=vdupq_n_s32(0),a3=vdupq_n_s32(0); int i=0;
            for(;i+64<=I;i+=64){
                uint8x16_t byo=vld1q_u8(wo+(i>>1)), byo1=vld1q_u8(wo1+(i>>1));
                uint8x16_t cyo=vld1q_u8(wo+(i>>1)+16), cyo1=vld1q_u8(wo1+(i>>1)+16);
                uint8x16x2_t zo =vzipq_u8(vandq_u8(byo, m4q), vshrq_n_u8(byo, 4));
                uint8x16x2_t zo1=vzipq_u8(vandq_u8(byo1,m4q), vshrq_n_u8(byo1,4));
                uint8x16x2_t ko =vzipq_u8(vandq_u8(cyo, m4q), vshrq_n_u8(cyo, 4));
                uint8x16x2_t ko1=vzipq_u8(vandq_u8(cyo1,m4q), vshrq_n_u8(cyo1,4));
                a0=mm_tile16(a0, vsubq_s8(vreinterpretq_s8_u8(zo.val[0]),b8q),
                                 vsubq_s8(vreinterpretq_s8_u8(zo1.val[0]),b8q),
                                 vld1q_s8(xs+i), vld1q_s8(xs1+i));
                a1=mm_tile16(a1, vsubq_s8(vreinterpretq_s8_u8(zo.val[1]),b8q),
                                 vsubq_s8(vreinterpretq_s8_u8(zo1.val[1]),b8q),
                                 vld1q_s8(xs+i+16), vld1q_s8(xs1+i+16));
                a2=mm_tile16(a2, vsubq_s8(vreinterpretq_s8_u8(ko.val[0]),b8q),
                                 vsubq_s8(vreinterpretq_s8_u8(ko1.val[0]),b8q),
                                 vld1q_s8(xs+i+32), vld1q_s8(xs1+i+32));
                a3=mm_tile16(a3, vsubq_s8(vreinterpretq_s8_u8(ko.val[1]),b8q),
                                 vsubq_s8(vreinterpretq_s8_u8(ko1.val[1]),b8q),
                                 vld1q_s8(xs+i+48), vld1q_s8(xs1+i+48));
            }
            for(;i+32<=I;i+=32){
                uint8x16_t byo=vld1q_u8(wo+(i>>1)), byo1=vld1q_u8(wo1+(i>>1));
                uint8x16x2_t zo =vzipq_u8(vandq_u8(byo, m4q), vshrq_n_u8(byo, 4));
                uint8x16x2_t zo1=vzipq_u8(vandq_u8(byo1,m4q), vshrq_n_u8(byo1,4));
                a0=mm_tile16(a0, vsubq_s8(vreinterpretq_s8_u8(zo.val[0]),b8q),
                                 vsubq_s8(vreinterpretq_s8_u8(zo1.val[0]),b8q),
                                 vld1q_s8(xs+i), vld1q_s8(xs1+i));
                a1=mm_tile16(a1, vsubq_s8(vreinterpretq_s8_u8(zo.val[1]),b8q),
                                 vsubq_s8(vreinterpretq_s8_u8(zo1.val[1]),b8q),
                                 vld1q_s8(xs+i+16), vld1q_s8(xs1+i+16));
            }
            int32x4_t acc=vaddq_s32(vaddq_s32(a0,a1),vaddq_s32(a2,a3));
            int32_t d00=vgetq_lane_s32(acc,0), d01=vgetq_lane_s32(acc,1);
            int32_t d10=vgetq_lane_s32(acc,2), d11=vgetq_lane_s32(acc,3);
            for(;i+1<I;i+=2){ uint8_t bo=wo[i>>1], bo1=wo1[i>>1];
                int a0=(int)(bo&0xF)-8, a1=(int)(bo>>4)-8, b0=(int)(bo1&0xF)-8, b1=(int)(bo1>>4)-8;
                int u0=xs[i],u1=xs[i+1],v0=xs1[i],v1=xs1[i+1];
                d00+=a0*u0+a1*u1; d01+=a0*v0+a1*v1; d10+=b0*u0+b1*u1; d11+=b0*v0+b1*v1; }
            if(i<I){ uint8_t bo=wo[i>>1], bo1=wo1[i>>1];
                int a0=(int)(bo&0xF)-8, b0=(int)(bo1&0xF)-8;
                d00+=a0*xs[i]; d01+=a0*xs1[i]; d10+=b0*xs[i]; d11+=b0*xs1[i]; }
            y[(int64_t)s*O+o]        =(float)d00*sc0*sx[s];
            y[(int64_t)s*O+(o+1)]    =(float)d10*sc1*sx[s];
            y[(int64_t)(s+1)*O+o]    =(float)d01*sc0*sx[s+1];
            y[(int64_t)(s+1)*O+(o+1)]=(float)d11*sc1*sx[s+1];
        }
        if(S&1){ int s=S-1; const int8_t *xs=xq+(int64_t)s*I;
            y[(int64_t)s*O+o]    =(float)dot_i4i8(wo, xs,I)*sc0*sx[s];
            y[(int64_t)s*O+(o+1)]=(float)dot_i4i8(wo1,xs,I)*sc1*sx[s]; }
    }
    if(O&1){ int o=O-1; const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        #pragma omp parallel for schedule(static)
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i4i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}
#endif

/* ---- IDOT dispatch (int8-quantized activations) --------------------------- */
static void matmul_q_idot(float *y, const int8_t *xq, const float *sx, const int8_t *q,
                          const float *scale, int S, int I, int O){
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    if(S>=2){ matmul_q_idot_mm(y,xq,sx,q,scale,S,I,O); return; }
#endif
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        for(int s0=0;s0<S;s0+=IDOT_MAX_ROWS){
            int n=S-s0<IDOT_MAX_ROWS?S-s0:IDOT_MAX_ROWS; int32_t d[IDOT_MAX_ROWS];
            if(n==1) d[0]=dot_i8i8(w,xq+(int64_t)s0*I,I);
            else dot_i8i8_rows(w,xq+(int64_t)s0*I,I,n,d);
            for(int r=0;r<n;r++) y[(int64_t)(s0+r)*O+o]=(float)d[r]*sc*sx[s0+r];
        } }
}
/* matmul_q_idot for one activation row (S = 1) and only the rows ids[0..n) of q: y[j] is
 * the output of row ids[j]. */
static void matmul_q_idot_rows(float *y, const int8_t *xq, float sx, const int8_t *q,
                               const float *scale, const int *ids, int n, int I){
    #pragma omp parallel for schedule(static)
    for(int j=0;j<n;j++) y[j]=(float)dot_i8i8(q+(int64_t)ids[j]*I,xq,I)*scale[ids[j]]*sx;
}
static void matmul_i4_idot(float *y, const int8_t *xq, const float *sx, const uint8_t *q4,
                           const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    if(S>=2){ matmul_i4_idot_mm(y,xq,sx,q4,scale,S,I,O); return; }
#endif
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i4i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}


/* ================= K1: layout int4 A PIANI (fmt 2/4, opzionale) ==============
 * Layout classico ("a coppie"): byte k = elementi (2k, 2k+1). Ogni kernel paga
 * unpacklo/unpackhi per riordinare i nibble, e l'IDOT paga sub+abs+sign per
 * forzare l'operando con segno dentro vpdpbusd (che e' nativamente u8 x s8).
 *
 * Layout A PIANI, per blocco di 64 elementi consecutivi (32 byte):
 *   byte k del blocco = (elem k + 8) | ((elem k+32 + 8) << 4)      k = 0..31
 * `and 0x0F` produce gli elementi 0..31 GIA' in ordine; `srli 4 + and` produce
 * 32..63 in ordine: l'unpack sparisce. La coda (I mod 64) resta a coppie.
 *
 * L'algebra dell'IDOT: i nibble memorizzati sono u = v+8 (unsigned 0..15), e
 *   dot(v, x) = dot(u, x) - 8 * sum(x)
 * dove sum(x) e' UN int32 per riga di attivazione, condiviso da ogni riga di
 * output. vpdpbusd consuma (u8, s8) direttamente: via sub+abs+sign (3 op
 * vettoriali per dpbusd). Misurato su questa famiglia: 1.87x L3-resident,
 * 25.3 GB/s DRAM (al tetto), 0 differenze di bit.
 * NB: allargare a 256 bit SENZA cambiare layout e' stato misurato 0.79-0.81x
 * (il fixup di lane costa piu' della larghezza): il layout e' la precondizione
 * della larghezza, non un'ottimizzazione indipendente.
 *
 * I kernel f32 planari preservano ESATTAMENTE l'ordine di accumulazione dei
 * gemelli a coppie (stesse lane, stessa sequenza 0..63): bit-identici anche
 * loro. EN: plane-nibble int4 layout. Low nibble of block byte k = element k,
 * high = element k+32; the unpack disappears, and the stored-unsigned nibbles
 * feed vpdpbusd natively via dot(v,x) = dot(u,x) - 8*sum(x). The f32 planar
 * kernels keep the pair kernels' exact accumulation order: bit-identical. */

/* riordina in place una riga (o un tensore [O,rb] riga per riga) da coppie a
 * piani; la coda I%64 resta a coppie. EN: in-place pair->planar repack. */
static void planarize_i4_row(uint8_t *row, int I){
    uint8_t tmp[32];
    int nb=I/64;
    for(int b=0;b<nb;b++){
        uint8_t *blk=row+b*32;
        /* pair: byte j ha (elem 2j, elem 2j+1). planar: byte k = (elem k, elem k+32) */
        for(int k=0;k<32;k++){
            int src_lo=k, src_hi=k+32;                       /* elementi voluti */
            uint8_t nib_lo=(blk[src_lo>>1]>>((src_lo&1)*4))&0xF;
            uint8_t nib_hi=(blk[src_hi>>1]>>((src_hi&1)*4))&0xF;
            tmp[k]=(uint8_t)(nib_lo|(nib_hi<<4));
        }
        memcpy(blk,tmp,32);
    }
}
static void planarize_i4(uint8_t *q4, int O, int I){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++) planarize_i4_row(q4+(int64_t)o*rb, I);
}

/* dot unsigned-nibble planare * int8: ritorna dot(u, x) SENZA la correzione
 * -8*sum(x) (la applica il chiamante, una volta per riga di output).
 * Coda I%64: nibble a coppie letti come unsigned, stessa identita'. */
static inline int32_t dot_i4p_u(const uint8_t *w4, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#if defined(__ARM_NEON)
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+64<=I;i+=64){
        uint8x16_t b0=vld1q_u8(w4+(i>>1)), b1=vld1q_u8(w4+(i>>1)+16);
        int8x16_t lo0=vreinterpretq_s8_u8(vandq_u8(b0,vdupq_n_u8(0x0F)));
        int8x16_t lo1=vreinterpretq_s8_u8(vandq_u8(b1,vdupq_n_u8(0x0F)));
        int8x16_t hi0=vreinterpretq_s8_u8(vshrq_n_u8(b0,4));
        int8x16_t hi1=vreinterpretq_s8_u8(vshrq_n_u8(b1,4));
#if defined(__ARM_FEATURE_DOTPROD)
        acc=vdotq_s32(acc,lo0,vld1q_s8(x+i));
        acc=vdotq_s32(acc,lo1,vld1q_s8(x+i+16));
        acc=vdotq_s32(acc,hi0,vld1q_s8(x+i+32));
        acc=vdotq_s32(acc,hi1,vld1q_s8(x+i+48));
#else
        int8x16_t xs0=vld1q_s8(x+i), xs1=vld1q_s8(x+i+16);
        int8x16_t xs2=vld1q_s8(x+i+32), xs3=vld1q_s8(x+i+48);
        int16x8_t m;
        m=vmull_s8(vget_low_s8(lo0),vget_low_s8(xs0));  acc=vpadalq_s16(acc,m);
        m=vmull_s8(vget_high_s8(lo0),vget_high_s8(xs0)); acc=vpadalq_s16(acc,m);
        m=vmull_s8(vget_low_s8(lo1),vget_low_s8(xs1));  acc=vpadalq_s16(acc,m);
        m=vmull_s8(vget_high_s8(lo1),vget_high_s8(xs1)); acc=vpadalq_s16(acc,m);
        m=vmull_s8(vget_low_s8(hi0),vget_low_s8(xs2));  acc=vpadalq_s16(acc,m);
        m=vmull_s8(vget_high_s8(hi0),vget_high_s8(xs2)); acc=vpadalq_s16(acc,m);
        m=vmull_s8(vget_low_s8(hi1),vget_low_s8(xs3));  acc=vpadalq_s16(acc,m);
        m=vmull_s8(vget_high_s8(hi1),vget_high_s8(xs3)); acc=vpadalq_s16(acc,m);
#endif
    }
    sum=vaddvq_s32(acc);
#endif
    for(;i+64<=I;i+=64){          /* fallback scalare sui blocchi planari */
        const uint8_t *blk=w4+(i>>1);
        for(int k=0;k<32;k++){
            sum+=(int32_t)(blk[k]&0xF)*x[i+k];
            sum+=(int32_t)(blk[k]>>4)*x[i+k+32];
        }
    }
    for(;i<I;i+=2){               /* coda a coppie, unsigned (u=v+8 memorizzato) */
        uint8_t byte=w4[i>>1];
        sum+=(int32_t)(byte&0xF)*x[i];
        if(i+1<I) sum+=(int32_t)(byte>>4)*x[i+1];
    }
    return sum;
}

/* vector body over an output range [o0,o1): the 1x4 row tile pays the weight
 * block's load+mask once per 4 activation rows. Integer group dots in any lane
 * order are exact, and each row keeps its own ascending-g fmaf chain. */
static void i4p_gidot_rows(float *y, const int8_t *xq, const float *sx,
                           const int32_t *xsg, const uint8_t *q4,
                           const float *scale, int S, int I, int O, int gs,
                           int o0, int o1){
    int rb=(I+1)/2, ng=(I+gs-1)/gs, bpg=gs/64;   /* blocchi-piano per gruppo */
    #pragma omp parallel for schedule(static)
    for(int o=o0;o<o1;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const float *scl=scale+(int64_t)o*ng;
        int s=0;
        for(; s+4<=S; s+=4){
            const int8_t  *x0=xq +(int64_t)s*I,  *x1=x0+I,  *x2=x1+I,  *x3=x2+I;
            const int32_t *g0=xsg+(int64_t)s*ng, *g1=g0+ng, *g2=g1+ng, *g3=g2+ng;
            float a0=0,a1=0,a2=0,a3=0;
            int g=0;
            for(; (g+1)*gs<=I; g++){
                int32_t d0,d1,d2,d3;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
                const uint8x16_t m4=vdupq_n_u8(0x0F);
                int32x4_t A0=vdupq_n_s32(0),A1=vdupq_n_s32(0),A2=vdupq_n_s32(0),A3=vdupq_n_s32(0);
                for(int b=0;b<bpg;b++){
                    int base=g*gs+b*64;
                    uint8x16_t b0=vld1q_u8(w+(base>>1)), b1=vld1q_u8(w+(base>>1)+16);
                    int8x16_t lo0=vreinterpretq_s8_u8(vandq_u8(b0,m4));
                    int8x16_t lo1=vreinterpretq_s8_u8(vandq_u8(b1,m4));
                    int8x16_t hi0=vreinterpretq_s8_u8(vshrq_n_u8(b0,4));
                    int8x16_t hi1=vreinterpretq_s8_u8(vshrq_n_u8(b1,4));
                    A0=vdotq_s32(A0,lo0,vld1q_s8(x0+base));    A0=vdotq_s32(A0,lo1,vld1q_s8(x0+base+16));
                    A0=vdotq_s32(A0,hi0,vld1q_s8(x0+base+32)); A0=vdotq_s32(A0,hi1,vld1q_s8(x0+base+48));
                    A1=vdotq_s32(A1,lo0,vld1q_s8(x1+base));    A1=vdotq_s32(A1,lo1,vld1q_s8(x1+base+16));
                    A1=vdotq_s32(A1,hi0,vld1q_s8(x1+base+32)); A1=vdotq_s32(A1,hi1,vld1q_s8(x1+base+48));
                    A2=vdotq_s32(A2,lo0,vld1q_s8(x2+base));    A2=vdotq_s32(A2,lo1,vld1q_s8(x2+base+16));
                    A2=vdotq_s32(A2,hi0,vld1q_s8(x2+base+32)); A2=vdotq_s32(A2,hi1,vld1q_s8(x2+base+48));
                    A3=vdotq_s32(A3,lo0,vld1q_s8(x3+base));    A3=vdotq_s32(A3,lo1,vld1q_s8(x3+base+16));
                    A3=vdotq_s32(A3,hi0,vld1q_s8(x3+base+32)); A3=vdotq_s32(A3,hi1,vld1q_s8(x3+base+48));
                }
                d0=vaddvq_s32(A0); d1=vaddvq_s32(A1);
                d2=vaddvq_s32(A2); d3=vaddvq_s32(A3);
#else
                d0=d1=d2=d3=0;
                for(int b=0;b<bpg;b++){
                    int base=g*gs+b*64;
                    const uint8_t *blk=w+(base>>1);
                    for(int k=0;k<32;k++){
                        int32_t ul=(int32_t)(blk[k]&0xF), uh=(int32_t)(blk[k]>>4);
                        d0+=ul*x0[base+k]+uh*x0[base+k+32];
                        d1+=ul*x1[base+k]+uh*x1[base+k+32];
                        d2+=ul*x2[base+k]+uh*x2[base+k+32];
                        d3+=ul*x3[base+k]+uh*x3[base+k+32];
                    }
                }
#endif
                a0=fmaf((float)(d0-8*g0[g]),scl[g],a0);
                a1=fmaf((float)(d1-8*g1[g]),scl[g],a1);
                a2=fmaf((float)(d2-8*g2[g]),scl[g],a2);
                a3=fmaf((float)(d3-8*g3[g]),scl[g],a3);
            }
            if(g*gs<I){                                  /* coda: gruppo parziale, nibble a coppie */
                int32_t d0=0,d1=0,d2=0,d3=0;
                for(int i=g*gs;i<I;i++){
                    uint8_t byte=w[i>>1];
                    int32_t u=(int32_t)((i&1)?(byte>>4):(byte&0xF));
                    d0+=u*x0[i]; d1+=u*x1[i]; d2+=u*x2[i]; d3+=u*x3[i];
                }
                a0=fmaf((float)(d0-8*g0[g]),scl[g],a0);
                a1=fmaf((float)(d1-8*g1[g]),scl[g],a1);
                a2=fmaf((float)(d2-8*g2[g]),scl[g],a2);
                a3=fmaf((float)(d3-8*g3[g]),scl[g],a3);
            }
            y[(int64_t)s*O+o]    =a0*sx[s];
            y[(int64_t)(s+1)*O+o]=a1*sx[s+1];
            y[(int64_t)(s+2)*O+o]=a2*sx[s+2];
            y[(int64_t)(s+3)*O+o]=a3*sx[s+3];
        }
        for(; s<S; s++){
            const int8_t *xr=xq+(int64_t)s*I;
            const int32_t *xg=xsg+(int64_t)s*ng;
            float a=0; int g=0;
            for(; (g+1)*gs<=I; g++){                     /* gruppi interi */
                int32_t d=0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
                const uint8x16_t m4=vdupq_n_u8(0x0F);
                int32x4_t acc=vdupq_n_s32(0);
                for(int b=0;b<bpg;b++){
                    int base=g*gs+b*64;
                    const uint8_t *blk=w+(base>>1);
                    const int8_t *xb=xr+base;
                    uint8x16_t b0=vld1q_u8(blk), b1=vld1q_u8(blk+16);
                    acc=vdotq_s32(acc,vreinterpretq_s8_u8(vandq_u8(b0,m4)),vld1q_s8(xb));
                    acc=vdotq_s32(acc,vreinterpretq_s8_u8(vandq_u8(b1,m4)),vld1q_s8(xb+16));
                    acc=vdotq_s32(acc,vreinterpretq_s8_u8(vshrq_n_u8(b0,4)),vld1q_s8(xb+32));
                    acc=vdotq_s32(acc,vreinterpretq_s8_u8(vshrq_n_u8(b1,4)),vld1q_s8(xb+48));
                }
                d=vaddvq_s32(acc);
#else
                for(int b=0;b<bpg;b++){
                    int base=g*gs+b*64;
                    const uint8_t *blk=w+(base>>1);
                    const int8_t *xb=xr+base;
                    for(int k=0;k<32;k++){
                        d+=(int32_t)(blk[k]&0xF)*xb[k];
                        d+=(int32_t)(blk[k]>>4)*xb[k+32];
                    }
                }
#endif
                a=fmaf((float)(d-8*xg[g]),scl[g],a);
            }
            if(g*gs<I){                                  /* coda: gruppo parziale, nibble a coppie */
                int32_t d=0;
                for(int i=g*gs;i<I;i++){
                    uint8_t byte=w[i>>1];
                    d+=(int32_t)((i&1)?(byte>>4):(byte&0xF))*xr[i];
                }
                a=fmaf((float)(d-8*xg[g]),scl[g],a);
            }
            y[(int64_t)s*O+o]=a*sx[s];
        }
    }
}

static void matmul_i4p_grouped_idot(float *y, const int8_t *xq, const float *sx,
                                    const int32_t *xsg, const uint8_t *q4,
                                    const float *scale, int S, int I, int O, int gs){
    i4p_gidot_rows(y,xq,sx,xsg,q4,scale,S,I,O,gs,0,O);
}

/* matmul IDOT planare (fmt=2): y = (dot_u - 8*xsum[s]) * scale[o] * sx[s].
 * Bit-identico a matmul_i4_idot: somme intere, identita' esatta. */
static void matmul_i4p_idot(float *y, const int8_t *xq, const float *sx, const int32_t *xsum,
                            const uint8_t *q4, const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        int s=0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        const uint8x16_t m4t=vdupq_n_u8(0x0F);
        for(;s+4<=S;s+=4){
            const int8_t *x0=xq+(int64_t)s*I, *x1=x0+I, *x2=x1+I, *x3=x2+I;
            int32x4_t a0=vdupq_n_s32(0), a1=vdupq_n_s32(0), a2=vdupq_n_s32(0), a3=vdupq_n_s32(0);
            int i=0;
            for(;i+64<=I;i+=64){
                uint8x16_t b0=vld1q_u8(w+(i>>1)), b1=vld1q_u8(w+(i>>1)+16);
                int8x16_t lo0=vreinterpretq_s8_u8(vandq_u8(b0,m4t));
                int8x16_t lo1=vreinterpretq_s8_u8(vandq_u8(b1,m4t));
                int8x16_t hi0=vreinterpretq_s8_u8(vshrq_n_u8(b0,4));
                int8x16_t hi1=vreinterpretq_s8_u8(vshrq_n_u8(b1,4));
                a0=vdotq_s32(a0,lo0,vld1q_s8(x0+i));    a0=vdotq_s32(a0,lo1,vld1q_s8(x0+i+16));
                a0=vdotq_s32(a0,hi0,vld1q_s8(x0+i+32)); a0=vdotq_s32(a0,hi1,vld1q_s8(x0+i+48));
                a1=vdotq_s32(a1,lo0,vld1q_s8(x1+i));    a1=vdotq_s32(a1,lo1,vld1q_s8(x1+i+16));
                a1=vdotq_s32(a1,hi0,vld1q_s8(x1+i+32)); a1=vdotq_s32(a1,hi1,vld1q_s8(x1+i+48));
                a2=vdotq_s32(a2,lo0,vld1q_s8(x2+i));    a2=vdotq_s32(a2,lo1,vld1q_s8(x2+i+16));
                a2=vdotq_s32(a2,hi0,vld1q_s8(x2+i+32)); a2=vdotq_s32(a2,hi1,vld1q_s8(x2+i+48));
                a3=vdotq_s32(a3,lo0,vld1q_s8(x3+i));    a3=vdotq_s32(a3,lo1,vld1q_s8(x3+i+16));
                a3=vdotq_s32(a3,hi0,vld1q_s8(x3+i+32)); a3=vdotq_s32(a3,hi1,vld1q_s8(x3+i+48));
            }
            int32_t d0=vaddvq_s32(a0), d1=vaddvq_s32(a1), d2=vaddvq_s32(a2), d3=vaddvq_s32(a3);
            for(;i<I;i+=2){
                uint8_t byte=w[i>>1];
                d0+=(int32_t)(byte&0xF)*x0[i]; d1+=(int32_t)(byte&0xF)*x1[i];
                d2+=(int32_t)(byte&0xF)*x2[i]; d3+=(int32_t)(byte&0xF)*x3[i];
                if(i+1<I){
                    d0+=(int32_t)(byte>>4)*x0[i+1]; d1+=(int32_t)(byte>>4)*x1[i+1];
                    d2+=(int32_t)(byte>>4)*x2[i+1]; d3+=(int32_t)(byte>>4)*x3[i+1];
                }
            }
            y[(int64_t)(s+0)*O+o]=(float)(d0-8*xsum[s+0])*sc*sx[s+0];
            y[(int64_t)(s+1)*O+o]=(float)(d1-8*xsum[s+1])*sc*sx[s+1];
            y[(int64_t)(s+2)*O+o]=(float)(d2-8*xsum[s+2])*sc*sx[s+2];
            y[(int64_t)(s+3)*O+o]=(float)(d3-8*xsum[s+3])*sc*sx[s+3];
        }
#endif
        for(;s<S;s++){
            int32_t d=dot_i4p_u(w,xq+(int64_t)s*I,I)-8*xsum[s];
            y[(int64_t)s*O+o]=(float)d*sc*sx[s];
        }
    }
}

#endif /* COLIBRI_IDOT_H */
