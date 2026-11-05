// Aegist GPU kernels. build.rs compiles this file to PTX; the engine loads it through the
// CUDA driver API. Storage: base weights int8 with a bf16 scale per output row, embeddings
// and small weights bf16, KV cache bf16. All arithmetic is fp32.
//
// bf16 is handled as raw 16-bit storage (no cuda_bf16.h): widening is a shift, narrowing
// rounds to nearest even, matching PyTorch.

typedef unsigned short bf16_t;

__device__ __forceinline__ float bf2f(bf16_t v) { return __uint_as_float(((unsigned int)v) << 16); }
__device__ __forceinline__ float bf_lo(unsigned int p) { return __uint_as_float(p << 16); }
__device__ __forceinline__ float bf_hi(unsigned int p) { return __uint_as_float(p & 0xffff0000u); }

__device__ __forceinline__ bf16_t f2bf(float f) {
    unsigned int u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16_t)0x7fc0;  // NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16_t)(u >> 16);
}

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024). Every thread gets the result.
__device__ float block_sum(float v, float* red) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) red[w] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    if (w == 0) v = warp_sum(v);
    if (threadIdx.x == 0) red[0] = v;
    __syncthreads();
    float r = red[0];
    __syncthreads();
    return r;
}

// 4 signed int8 packed in an int, dotted with 4 floats.
__device__ __forceinline__ float dot4_i8(int w, float4 a) {
    return a.x * (float)((w << 24) >> 24) + a.y * (float)((w << 16) >> 24) +
           a.z * (float)((w << 8) >> 24) + a.w * (float)(w >> 24);
}

__device__ __forceinline__ float silu(float a) { return a / (1.f + expf(-a)); }

// One warp: dot(x, W[n, :]) for an int8 row of K entries (K % 16 == 0).
__device__ __forceinline__ float row_dot_i8(const float* __restrict__ x, const signed char* __restrict__ W, int n, int K,
                                            int lane) {
    const int4* row = (const int4*)(W + (size_t)n * K);
    const float4* xv = (const float4*)x;
    float acc = 0.f;
    for (int c = lane; c < K / 16; c += 32) {
        int4 p = __ldg(row + c);
        acc += dot4_i8(p.x, xv[4 * c]) + dot4_i8(p.y, xv[4 * c + 1]) + dot4_i8(p.z, xv[4 * c + 2]) +
               dot4_i8(p.w, xv[4 * c + 3]);
    }
    return warp_sum(acc);
}

extern "C" __global__ void embed(const bf16_t* __restrict__ table, const int* __restrict__ ids, float* __restrict__ out,
                                 int D) {
    const bf16_t* row = table + (size_t)ids[blockIdx.x] * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[(size_t)blockIdx.x * D + i] = bf2f(row[i]);
}

// out = w * x / rms(x), one block per row.
extern "C" __global__ void rmsnorm(const float* __restrict__ x, const bf16_t* __restrict__ w, float* __restrict__ out,
                                   int D, float eps) {
    __shared__ float red[32];
    const float* xr = x + (size_t)blockIdx.x * D;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, red);
    float inv = rsqrtf(ss / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[(size_t)blockIdx.x * D + i] = bf2f(w[i]) * (xr[i] * inv);
}

// Decode matvec: y[n] (+)= scale[n] * dot(x, W[n]). One warp per row.
extern "C" __global__ void gemv_i8(const float* __restrict__ x, const signed char* __restrict__ W,
                                   const bf16_t* __restrict__ scale, float* __restrict__ y, int N, int K, int accumulate) {
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (n >= N) return;
    float v = row_dot_i8(x, W, n, K, lane) * bf2f(scale[n]);
    if (lane == 0) y[n] = accumulate ? y[n] + v : v;
}

// Decode SwiGLU input side: y[n] = silu(gate_n . x) * (up_n . x), both rows read by one warp.
extern "C" __global__ void gemv_i8_gateup(const float* __restrict__ x, const signed char* __restrict__ Wg,
                                          const bf16_t* __restrict__ sg, const signed char* __restrict__ Wu,
                                          const bf16_t* __restrict__ su, float* __restrict__ y, int N, int K) {
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (n >= N) return;
    float g = row_dot_i8(x, Wg, n, K, lane) * bf2f(sg[n]);
    float u = row_dot_i8(x, Wu, n, K, lane) * bf2f(su[n]);
    if (lane == 0) y[n] = silu(g) * u;
}

// y[n] = dot(x, W[n]) for a bf16 matrix (the tied output head). K % 8 == 0.
extern "C" __global__ void gemv_bf16(const float* __restrict__ x, const bf16_t* __restrict__ W, float* __restrict__ y,
                                     int N, int K) {
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (n >= N) return;
    const uint4* row = (const uint4*)(W + (size_t)n * K);
    const float4* xv = (const float4*)x;
    float acc = 0.f;
    for (int c = lane; c < K / 8; c += 32) {
        uint4 p = __ldg(row + c);
        float4 a = xv[2 * c], b = xv[2 * c + 1];
        acc += a.x * bf_lo(p.x) + a.y * bf_hi(p.x) + a.z * bf_lo(p.y) + a.w * bf_hi(p.y) + b.x * bf_lo(p.z) +
               b.y * bf_hi(p.z) + b.z * bf_lo(p.w) + b.w * bf_hi(p.w);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[n] = acc;
}

// Prefill matmul: Y[m, n] (+)= scale[n] * sum_k X[m, k] W[n, k]. Tiles of 32 tokens x 64 outputs,
// 256 threads, each computing 2 x 4 outputs. K % 32 == 0.
#define BM 32
#define BN 64
#define BK 32
extern "C" __global__ void gemm_i8(const float* __restrict__ X, const signed char* __restrict__ W,
                                   const bf16_t* __restrict__ scale, float* __restrict__ Y, int M, int N, int K,
                                   int accumulate) {
    __shared__ float Xs[BM][BK + 1];
    __shared__ float Ws[BN][BK + 1];
    int tid = threadIdx.x, ty = tid / 16, tx = tid % 16;
    int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * BK; i += 256) {
            int r = i / BK, c = i % BK, m = m0 + r;
            Xs[r][c] = m < M ? X[(size_t)m * K + k0 + c] : 0.f;
        }
        for (int i = tid; i < BN * BK; i += 256) {
            int r = i / BK, c = i % BK, n = n0 + r;
            Ws[r][c] = n < N ? (float)W[(size_t)n * K + k0 + c] : 0.f;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < BK; kk++) {
            float a0 = Xs[ty * 2][kk], a1 = Xs[ty * 2 + 1][kk];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float b = Ws[tx * 4 + j][kk];
                acc[0][j] += a0 * b;
                acc[1][j] += a1 * b;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int m = m0 + ty * 2 + i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int n = n0 + tx * 4 + j;
            if (n >= N) continue;
            float v = acc[i][j] * bf2f(scale[n]);
            size_t o = (size_t)m * N + n;
            Y[o] = accumulate ? Y[o] + v : v;
        }
    }
}

extern "C" __global__ void silu_mul(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ out,
                                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = silu(g[i]) * u[i];
}

// Per-head RMSNorm (QK-norm) then rotary embedding (rotate-half form), one block of HD threads per
// (token, head). Query heads are rewritten in place; key heads, and the matching value heads, go
// into this layer's cache at position pos0 + token. Cache layout: [KV, cap, HD].
extern "C" __global__ void qk_norm_rope(float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                                        const bf16_t* __restrict__ qw, const bf16_t* __restrict__ kw,
                                        bf16_t* __restrict__ kc, bf16_t* __restrict__ vc, int H, int KV, int HD,
                                        int pos0, int cap, float eps, float theta) {
    __shared__ float red[32];
    __shared__ float buf[256];
    int t = blockIdx.x, hh = blockIdx.y, i = threadIdx.x;
    bool isq = hh < H;
    int h = isq ? hh : hh - H;
    const float* src = isq ? q + ((size_t)t * H + h) * HD : k + ((size_t)t * KV + h) * HD;
    float xv = src[i];
    float ss = block_sum(xv * xv, red);
    float nv = bf2f(isq ? qw[i] : kw[i]) * (xv * rsqrtf(ss / HD + eps));
    buf[i] = nv;
    __syncthreads();
    int half = HD / 2, j = i < half ? i : i - half;
    float inv = 1.0f / powf(theta, (float)(2 * j) / (float)HD);
    float ang = (float)(pos0 + t) * inv;
    float c = cosf(ang), s = sinf(ang);
    float out = i < half ? nv * c - buf[i + half] * s : nv * c + buf[i - half] * s;
    if (isq) {
        q[((size_t)t * H + h) * HD + i] = out;
    } else {
        size_t o = ((size_t)h * cap + pos0 + t) * HD + i;
        kc[o] = f2bf(out);
        vc[o] = f2bf(v[((size_t)t * KV + h) * HD + i]);
    }
}

// Causal grouped-query attention with an online softmax (no score matrix is stored).
// One block of 8 warps per (token, query head); warps stride over the keys, each lane owns 4 of
// the 128 head dimensions, and the warps' partial results are merged at the end.
extern "C" __global__ void attention(const float* __restrict__ q, const bf16_t* __restrict__ kc,
                                     const bf16_t* __restrict__ vc, float* __restrict__ out, int H, int KV, int pos0,
                                     int cap, float scale) {
    const int HD = 128, NW = 8;
    __shared__ float sm[NW], ssum[NW];
    __shared__ float sacc[NW][HD];
    int t = blockIdx.x, h = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int kvh = h / (H / KV);
    float4 qv = ((const float4*)(q + ((size_t)t * H + h) * HD))[lane];
    int L = pos0 + t + 1;
    const bf16_t* kb = kc + (size_t)kvh * cap * HD;
    const bf16_t* vb = vc + (size_t)kvh * cap * HD;
    float m = neg_inf(), s = 0.f, a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int l = warp; l < L; l += NW) {
        uint2 kp = ((const uint2*)(kb + (size_t)l * HD))[lane];
        float d = warp_sum(qv.x * bf_lo(kp.x) + qv.y * bf_hi(kp.x) + qv.z * bf_lo(kp.y) + qv.w * bf_hi(kp.y)) * scale;
        float mn = fmaxf(m, d);
        float corr = expf(m - mn), p = expf(d - mn);
        uint2 vp = ((const uint2*)(vb + (size_t)l * HD))[lane];
        s = s * corr + p;
        a0 = a0 * corr + p * bf_lo(vp.x);
        a1 = a1 * corr + p * bf_hi(vp.x);
        a2 = a2 * corr + p * bf_lo(vp.y);
        a3 = a3 * corr + p * bf_hi(vp.y);
        m = mn;
    }
    if (lane == 0) {
        sm[warp] = m;
        ssum[warp] = s;
    }
    sacc[warp][lane * 4] = a0;
    sacc[warp][lane * 4 + 1] = a1;
    sacc[warp][lane * 4 + 2] = a2;
    sacc[warp][lane * 4 + 3] = a3;
    __syncthreads();
    if (threadIdx.x < HD) {
        float M = neg_inf();
        for (int w = 0; w < NW; w++) M = fmaxf(M, sm[w]);
        float S = 0.f, A = 0.f;
        for (int w = 0; w < NW; w++) {
            if (sm[w] == neg_inf()) continue;  // this warp saw no keys
            float f = expf(sm[w] - M);
            S += ssum[w] * f;
            A += sacc[w][threadIdx.x] * f;
        }
        out[((size_t)t * H + h) * HD + threadIdx.x] = A / S;
    }
}

// Indices and values of the k largest of v[0..n), largest first (ties: lower index first).
__device__ void topk_small(const float* v, int n, int k, float* ov, int* oi) {
    for (int j = 0; j < k; j++) {
        int best = -1;
        for (int i = 0; i < n; i++) {
            bool taken = false;
            for (int p = 0; p < j; p++) taken |= oi[p] == i;
            if (!taken && (best < 0 || v[i] > v[best])) best = i;
        }
        oi[j] = best;
        ov[j] = v[best];
    }
}

// The core engine and the switchable cores, one block (256 threads) per token.
// Routing: product keys (two score tables of n1 and n2 entries address n1*n2 cores), top-k of the
// k1*k2 candidate pairs, softmax weights. Then each chosen core that is live (has a slot in the
// compact live bank) adds  w * down^T (silu(h . gate) * (h . up)). A silent core adds exactly
// zero, so it is skipped. counts[core] tallies switch-ons for the launcher's stats.
extern "C" __global__ void cores_apply(const float* __restrict__ h, const bf16_t* __restrict__ key1,
                                       const bf16_t* __restrict__ key2, const int* __restrict__ live,
                                       const bf16_t* __restrict__ cg, const bf16_t* __restrict__ cu,
                                       const bf16_t* __restrict__ cd, float* __restrict__ x, int* __restrict__ counts,
                                       int D, int n1, int n2, int k, int r) {
    __shared__ float s1[128], s2[64], z[64], selw[8];
    __shared__ int sel[8];
    int t = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const float* hr = h + (size_t)t * D;
    for (int j = warp; j < n1 + n2; j += nw) {
        const bf16_t* w = j < n1 ? key1 + (size_t)j * D : key2 + (size_t)(j - n1) * D;
        float a = 0.f;
        for (int i = lane; i < D; i += 32) a += hr[i] * bf2f(w[i]);
        a = warp_sum(a);
        if (lane == 0) {
            if (j < n1) s1[j] = a;
            else s2[j - n1] = a;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int k1 = min(k, n1), k2 = min(k, n2), i1[8], i2[8], ci[64], ti[8];
        float v1[8], v2[8], cv[64], tv[8];
        topk_small(s1, n1, k1, v1, i1);
        topk_small(s2, n2, k2, v2, i2);
        int nc = 0;
        for (int a = 0; a < k1; a++)
            for (int b = 0; b < k2; b++) {
                cv[nc] = v1[a] + v2[b];
                ci[nc++] = i1[a] * n2 + i2[b];
            }
        topk_small(cv, nc, k, tv, ti);
        float sum = 0.f;
        for (int j = 0; j < k; j++) sum += expf(tv[j] - tv[0]);
        for (int j = 0; j < k; j++) {
            sel[j] = ci[ti[j]];
            selw[j] = expf(tv[j] - tv[0]) / sum;
            atomicAdd(counts + sel[j], 1);
        }
    }
    __syncthreads();
    for (int c = 0; c < k; c++) {
        int li = live[sel[c]];
        if (li < 0) continue;  // same for the whole block
        const bf16_t* G = cg + (size_t)li * D * r;
        const bf16_t* U = cu + (size_t)li * D * r;
        const bf16_t* Dn = cd + (size_t)li * r * D;
        for (int j = warp; j < r; j += nw) {
            float a = 0.f, b = 0.f;
            for (int i = lane; i < D; i += 32) {
                float hv = hr[i];
                a += hv * bf2f(G[(size_t)i * r + j]);
                b += hv * bf2f(U[(size_t)i * r + j]);
            }
            a = warp_sum(a);
            b = warp_sum(b);
            if (lane == 0) z[j] = silu(a) * b;
        }
        __syncthreads();
        float wc = selw[c];
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            float acc = 0.f;
            for (int j = 0; j < r; j++) acc += z[j] * bf2f(Dn[(size_t)j * D + i]);
            x[(size_t)t * D + i] += wc * acc;
        }
        __syncthreads();
    }
}

#include "hybrid.cu"
#include "next.cu"
#include "max.cu"
#include "kimi.cu"
