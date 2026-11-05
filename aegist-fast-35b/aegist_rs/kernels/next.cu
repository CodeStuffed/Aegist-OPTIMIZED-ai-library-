// Kernels for the second language core (hyper-connection lanes, per-layer n-gram embeddings,
// the QSA sparse-attention indexer, FP8 experts, bf16 linears). Included after hybrid.cu, so it
// shares its helpers. Same conventions: fp32 activations, weights widened on the fly.

// float8 e4m3 (fn) byte -> float, exact for normals and subnormals: the 7 low bits shifted into an
// fp32's exponent/mantissa give the value times 2^-120 (a denormal for e4m3 subnormals), and the
// multiply by 2^120 is exact. Needs denormals kept (no --use_fast_math / ftz).
__device__ __forceinline__ float e4m3(unsigned int b) {
    return __uint_as_float(((b & 0x80u) << 24) | ((b & 0x7fu) << 20)) * __int_as_float(247 << 23);
}

// 4 e4m3 codes (byte j = element j) dotted with 4 floats.
__device__ __forceinline__ float dot4_f8(unsigned int p, float4 a) {
    return a.x * e4m3(p & 255u) + a.y * e4m3((p >> 8) & 255u) + a.z * e4m3((p >> 16) & 255u) + a.w * e4m3(p >> 24);
}

__device__ __forceinline__ float dot16_f8(uint4 w, const float4* x) {
    return dot4_f8(w.x, x[0]) + dot4_f8(w.y, x[1]) + dot4_f8(w.z, x[2]) + dot4_f8(w.w, x[3]);
}

// --- bf16 linears ----------------------------------------------------------------------------

// Decode matvec: y[n] (+)= dot(x, W[n]) for bf16 W [N, K], K % 8 == 0. One warp per row.
extern "C" __global__ void gemv_bf16a(const float* __restrict__ x, const bf16_t* __restrict__ W, float* __restrict__ y,
                                      int N, int K, int accumulate) {
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
    if (lane == 0) y[n] = accumulate ? y[n] + acc : acc;
}

// Prefill matmul: Y[m, n] (+)= sum_k X[m, k] W[n, k], bf16 W. Tiles of 32 x 64, K % 32 == 0.
extern "C" __global__ void gemm_bf16(const float* __restrict__ X, const bf16_t* __restrict__ W, float* __restrict__ Y,
                                     int M, int N, int K, int accumulate) {
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
            Ws[r][c] = n < N ? bf2f(W[(size_t)n * K + k0 + c]) : 0.f;
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
            size_t o = (size_t)m * N + n;
            Y[o] = accumulate ? Y[o] + acc[i][j] : acc[i][j];
        }
    }
}

// --- hyper-connection lanes --------------------------------------------------------------------

// Zero-centred RMSNorm of each D-wide group of a row of G*D values, weight [G*D]. Grid rows*G.
extern "C" __global__ void gznorm(const float* __restrict__ x, const bf16_t* __restrict__ w, float* __restrict__ out,
                                  int D, int G, float eps) {
    __shared__ float red[32];
    size_t base = (size_t)blockIdx.x * D;
    const bf16_t* wg = w + (size_t)(blockIdx.x % G) * D;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += x[base + i] * x[base + i];
    ss = block_sum(ss, red);
    float inv = rsqrtf(ss / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[base + i] = (1.f + bf2f(wg[i])) * (x[base + i] * inv);
}

// x[i] = silu(x[i] * s)
extern "C" __global__ void scale_silu(float* __restrict__ x, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = silu(x[i] * s);
}

// The sublayer input: h[t, i] = mean over lanes l of sigmoid(wl[t, l, i]) * xn[t, l, i].
// Grid (T, ceil(D / 256)), block 256.
extern "C" __global__ void hc_mix(const float* __restrict__ wl, const float* __restrict__ xn, float* __restrict__ h, int D,
                                  int HC) {
    int t = blockIdx.x, i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= D) return;
    size_t r = (size_t)t * HC * D + i;
    float s = 0.f;
    for (int l = 0; l < HC; l++) s += sigmoidf_(wl[r + (size_t)l * D]) * xn[r + (size_t)l * D];
    h[(size_t)t * D + i] = s / HC;
}

// The sublayer output into every lane: x[t, l, i] += y[t, i] * 2 sigmoid(il[t, l] / HC).
extern "C" __global__ void hc_inject(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ il, int D,
                                     int HC) {
    int t = blockIdx.x, i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= D) return;
    float yv = y[(size_t)t * D + i];
    for (int l = 0; l < HC; l++)
        x[((size_t)t * HC + l) * D + i] += yv * 2.f * sigmoidf_(il[(size_t)t * HC + l] / HC);
}

// --- DeltaNet output norm with a choice of gate ---------------------------------------------------

// out = w * rmsnorm(o) * act(z) per value head, act = sigmoid (sig != 0) or SiLU. Grid (T, HV), block DV.
extern "C" __global__ void gated_norm2(const float* __restrict__ o, const float* __restrict__ z,
                                       const bf16_t* __restrict__ w, float* __restrict__ out, int HV, int DV, float eps,
                                       int sig) {
    __shared__ float red[32];
    size_t base = ((size_t)blockIdx.x * HV + blockIdx.y) * DV;
    int i = threadIdx.x;
    float v = o[base + i];
    float ss = block_sum(v * v, red);
    float zv = z[base + i];
    out[base + i] = bf2f(w[i]) * (v * rsqrtf(ss / DV + eps)) * (sig ? sigmoidf_(zv) : silu(zv));
}

// --- router for up to 1024 experts, top 16 ----------------------------------------------------

// As router_topk, for E <= blockDim.x <= 1024 and K <= 16. One block per token.
extern "C" __global__ void router_topk_wide(const float* __restrict__ h, const bf16_t* __restrict__ W, int* __restrict__ idx,
                                            float* __restrict__ wts, int D, int E, int K) {
    __shared__ float lg[1024], red[32], rv[32], sv[16];
    __shared__ int ri[32], si[16];
    int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nw = blockDim.x >> 5;
    const float4* xv = (const float4*)(h + (size_t)t * D);
    for (int e = warp; e < E; e += nw) {
        const uint4* row = (const uint4*)(W + (size_t)e * D);
        float acc = 0.f;
        for (int c = lane; c < D / 8; c += 32) {
            uint4 p = __ldg(row + c);
            float4 a = xv[2 * c], b = xv[2 * c + 1];
            acc += a.x * bf_lo(p.x) + a.y * bf_hi(p.x) + a.z * bf_lo(p.y) + a.w * bf_hi(p.y) + b.x * bf_lo(p.z) +
                   b.y * bf_hi(p.z) + b.z * bf_lo(p.w) + b.w * bf_hi(p.w);
        }
        acc = warp_sum(acc);
        if (lane == 0) lg[e] = acc;
    }
    __syncthreads();
    float v = tid < E ? lg[tid] : neg_inf();
    float M = block_max(v, red);
    float p = tid < E ? expf(v - M) : 0.f;
    float S = block_sum(p, red);
    float mine = tid < E ? p / S : -1.f;
    for (int k = 0; k < K; k++) {
        float bv;
        int bi;
        block_argmax(mine, tid, rv, ri, &bv, &bi);
        if (tid == 0) {
            sv[k] = bv;
            si[k] = bi;
        }
        if (tid == bi) mine = -1.f;
    }
    __syncthreads();
    if (tid == 0) {
        float sum = 0.f;
        for (int k = 0; k < K; k++) sum += sv[k];
        for (int k = 0; k < K; k++) {
            idx[(size_t)t * K + k] = si[k];
            wts[(size_t)t * K + k] = sv[k] / sum;
        }
    }
}

// --- FP8 experts -------------------------------------------------------------------------------
// One expert's bytes (aegist/hc.py expert_layout_fp8): gate_up e4m3 [2I, D] (gate rows, then up
// rows), down e4m3 [D, I], then bf16 scales gu_s [2I/128, ceil(D/128)] at offs and dn_s
// [ceil(D/128), ceil(I/128)] at offd. Weight = code * scale of its 128 x 128 block. Picks are grouped
// by expert as for the 4-bit kernels (eptr, ustart, ptok, psrc). D % 16 == 0, I % 128 == 0.

// SwiGLU input side for few picks per expert. One warp per unit r; grid (I/8, U), 256 threads.
extern "C" __global__ void moe8_gu_gemv(const float* __restrict__ h, const unsigned long long* __restrict__ eptr,
                                        const int* __restrict__ ustart, const int* __restrict__ ptok,
                                        float* __restrict__ act, int D, int I, int offs) {
    int lane = threadIdx.x & 31, r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), u = blockIdx.y;
    if (r >= I) return;
    const unsigned char* base = (const unsigned char*)eptr[u];
    const uint4* qg = (const uint4*)(base + (size_t)r * D);
    const uint4* qu = (const uint4*)(base + (size_t)(I + r) * D);
    int SB = (D + 127) / 128;
    const bf16_t* sg = (const bf16_t*)(base + offs) + (size_t)(r >> 7) * SB;
    const bf16_t* su = (const bf16_t*)(base + offs) + (size_t)((I + r) >> 7) * SB;
    for (int p = ustart[u]; p < ustart[u + 1]; p++) {
        const float4* xv = (const float4*)(h + (size_t)ptok[p] * D);
        float ag = 0.f, au = 0.f;
        for (int c = lane; c < D / 16; c += 32) {
            uint4 wg = __ldg(qg + c), wu = __ldg(qu + c);
            ag += dot16_f8(wg, xv + 4 * c) * bf2f(sg[c >> 3]);
            au += dot16_f8(wu, xv + 4 * c) * bf2f(su[c >> 3]);
        }
        ag = warp_sum(ag);
        au = warp_sum(au);
        if (lane == 0) act[(size_t)p * I + r] = silu(ag) * au;
    }
}

// SwiGLU output side for few picks per expert: ypick[psrc[p], o] = wts[psrc[p]] * (down_o . act[p]).
// One warp per output row; grid (ceil(D/8), U), 256 threads.
extern "C" __global__ void moe8_dn_gemv(const float* __restrict__ act, const unsigned long long* __restrict__ eptr,
                                        const int* __restrict__ ustart, const int* __restrict__ psrc,
                                        const float* __restrict__ wts, float* __restrict__ ypick, int D, int I, int offq,
                                        int offd) {
    int lane = threadIdx.x & 31, o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), u = blockIdx.y;
    if (o >= D) return;
    const unsigned char* base = (const unsigned char*)eptr[u];
    const uint4* q = (const uint4*)(base + offq + (size_t)o * I);
    const bf16_t* s = (const bf16_t*)(base + offd) + (size_t)(o >> 7) * ((I + 127) / 128);
    for (int p = ustart[u]; p < ustart[u + 1]; p++) {
        const float4* a = (const float4*)(act + (size_t)p * I);
        float acc = 0.f;
        for (int c = lane; c < I / 16; c += 32) acc += dot16_f8(__ldg(q + c), a + 4 * c) * bf2f(s[c >> 3]);
        acc = warp_sum(acc);
        if (lane == 0) {
            int src = psrc[p];
            ypick[(size_t)src * D + o] = acc * wts[src];
        }
    }
}

// Rows [row0, row0 + n) x columns [col0, col0 + 64) of an e4m3 matrix (row stride rs bytes, scale
// grid SB blocks per row) into Ws[n][65] as floats. col0 % 64 == 0, so the tile is in one scale block.
__device__ __forceinline__ void load_f8_tile(float (*Ws)[65], const unsigned char* q, const bf16_t* sc, int row0, int n,
                                             int rs, int SB, int col0) {
    for (int i = threadIdx.x; i < n * 16; i += blockDim.x) {
        int r = i >> 4, c = i & 15, row = row0 + r;
        unsigned int p = *(const unsigned int*)(q + (size_t)row * rs + col0 + c * 4);
        float s = bf2f(sc[(size_t)(row >> 7) * SB + (col0 >> 7)]);
#pragma unroll
        for (int b = 0; b < 4; b++) Ws[r][c * 4 + b] = e4m3((p >> (8 * b)) & 255u) * s;
    }
}

// SwiGLU input side for many picks per expert (prefill). Grid (I/32, U), 256 threads. D % 64 == 0.
extern "C" __global__ void moe8_gu_gemm(const float* __restrict__ h, const unsigned long long* __restrict__ eptr,
                                        const int* __restrict__ ustart, const int* __restrict__ ptok,
                                        float* __restrict__ act, int D, int I, int offs) {
    __shared__ float Xs[32][65];
    __shared__ float Ws[64][65];
    int u = blockIdx.y, r0 = blockIdx.x * 32, tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const unsigned char* base = (const unsigned char*)eptr[u];
    const bf16_t* sc = (const bf16_t*)(base + offs);
    int SB = (D + 127) / 128, p_end = ustart[u + 1];
    for (int p0 = ustart[u]; p0 < p_end; p0 += 32) {
        float ag[2][2] = {{0.f, 0.f}, {0.f, 0.f}}, au[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
        for (int col0 = 0; col0 < D; col0 += 64) {
            for (int i = tid; i < 32 * 64; i += blockDim.x) {
                int pr = i >> 6, c = i & 63, p = p0 + pr;
                Xs[pr][c] = p < p_end ? h[(size_t)ptok[p] * D + col0 + c] : 0.f;
            }
            load_f8_tile(Ws, base, sc, r0, 32, D, SB, col0);
            load_f8_tile(Ws + 32, base, sc, I + r0, 32, D, SB, col0);
            __syncthreads();
#pragma unroll 8
            for (int kk = 0; kk < 64; kk++) {
                float a0 = Xs[ty * 2][kk], a1 = Xs[ty * 2 + 1][kk];
                float g0 = Ws[tx * 2][kk], g1 = Ws[tx * 2 + 1][kk], u0 = Ws[32 + tx * 2][kk], u1 = Ws[33 + tx * 2][kk];
                ag[0][0] += a0 * g0;
                ag[0][1] += a0 * g1;
                ag[1][0] += a1 * g0;
                ag[1][1] += a1 * g1;
                au[0][0] += a0 * u0;
                au[0][1] += a0 * u1;
                au[1][0] += a1 * u0;
                au[1][1] += a1 * u1;
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int p = p0 + ty * 2 + i;
            if (p >= p_end) continue;
#pragma unroll
            for (int j = 0; j < 2; j++) act[(size_t)p * I + r0 + tx * 2 + j] = silu(ag[i][j]) * au[i][j];
        }
    }
}

// SwiGLU output side for many picks per expert (prefill). Grid (D/64, U), 256 threads. I % 64 == 0.
extern "C" __global__ void moe8_dn_gemm(const float* __restrict__ act, const unsigned long long* __restrict__ eptr,
                                        const int* __restrict__ ustart, const int* __restrict__ psrc,
                                        const float* __restrict__ wts, float* __restrict__ ypick, int D, int I, int offq,
                                        int offd) {
    __shared__ float Xs[32][65];
    __shared__ float Ws[64][65];
    int u = blockIdx.y, o0 = blockIdx.x * 64, tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const unsigned char* base = (const unsigned char*)eptr[u];
    const bf16_t* sc = (const bf16_t*)(base + offd);
    int SB = (I + 127) / 128, p_end = ustart[u + 1];
    for (int p0 = ustart[u]; p0 < p_end; p0 += 32) {
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int col0 = 0; col0 < I; col0 += 64) {
            for (int i = tid; i < 32 * 64; i += blockDim.x) {
                int pr = i >> 6, c = i & 63, p = p0 + pr;
                Xs[pr][c] = p < p_end ? act[(size_t)p * I + col0 + c] : 0.f;
            }
            load_f8_tile(Ws, base + offq, sc, o0, 64, I, SB, col0);
            __syncthreads();
#pragma unroll 8
            for (int kk = 0; kk < 64; kk++) {
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
            int p = p0 + ty * 2 + i;
            if (p >= p_end) continue;
            int src = psrc[p];
            float wv = wts[src];
#pragma unroll
            for (int j = 0; j < 4; j++) ypick[(size_t)src * D + o0 + tx * 4 + j] = acc[i][j] * wv;
        }
    }
}

// --- 4-bit copies of FP8 experts ($AEGIST_EXPERT_BITS=4) -------------------------------------
// The pool can hold experts as 4-bit copies, nearly twice as many in the same VRAM. Layout as in
// hybrid.cu (aegist/hybrid.py expert_layout): gate_up codes [2I][D/2] at 0, f16 scale and min
// [2I][D/64] at s4g / m4g, down codes [D][I/2] at q4d, scale and min [D][I/64] at s4d / m4d;
// weight = code * scale + min over each 64 values of a row. moe_gu_gemv, moe_gu_gemm and
// moe_dn_gemm run them as they are; moe4_dn_gemv below takes any I (moe_dn_gemv needs 512).

__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

// One FP8 expert at src (layout of the moe8 kernels) as a 4-bit expert at dst: each 64-value group
// of a row gets min = its smallest weight and scale = its range / 15. One warp per group, 256 threads.
extern "C" __global__ void q4_from_f8(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int D,
                                      int I, int offs, int offq, int offd, int s4g, int m4g, int q4d, int s4d, int m4d) {
    int lane = threadIdx.x & 31;
    long long w = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    long long ngu = (long long)2 * I * (D / 64), ndn = (long long)D * (I / 64);
    if (w >= ngu + ndn) return;
    bool gu = w < ngu;
    int cols = gu ? D : I, G = cols / 64, SB = (cols + 127) / 128;
    long long ww = gu ? w : w - ngu;
    int row = (int)(ww / G), g = (int)(ww % G);
    const unsigned char* q = gu ? src : src + offq;
    float s8 = bf2f(((const bf16_t*)(src + (gu ? offs : offd)))[(size_t)(row >> 7) * SB + (g >> 1)]);
    unsigned int pr = *(const unsigned short*)(q + (size_t)row * cols + g * 64 + lane * 2);
    float a = e4m3(pr & 255u) * s8, b = e4m3(pr >> 8) * s8;
    float lo = fminf(a, b), hi = fmaxf(a, b);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        lo = fminf(lo, __shfl_xor_sync(0xffffffffu, lo, o));
        hi = fmaxf(hi, __shfl_xor_sync(0xffffffffu, hi, o));
    }
    unsigned short hs = f2h((hi - lo) / 15.f), hm = f2h(lo);
    float s = h2f(hs), m = h2f(hm), inv = s > 0.f ? 1.f / s : 0.f;
    int ca = min(15, max(0, __float2int_rn((a - m) * inv))), cb = min(15, max(0, __float2int_rn((b - m) * inv)));
    dst[(gu ? 0 : (size_t)q4d) + (size_t)row * (cols / 2) + g * 32 + lane] = (unsigned char)(ca | (cb << 4));
    if (lane == 0) {
        ((unsigned short*)(dst + (gu ? s4g : s4d)))[(size_t)row * G + g] = hs;
        ((unsigned short*)(dst + (gu ? m4g : m4d)))[(size_t)row * G + g] = hm;
    }
}

// SwiGLU output side of 4-bit experts for few picks per expert: ypick[psrc[p], o] = wts[psrc[p]] *
// (down_o . act[p]). One warp per output row; grid (ceil(D/8), U), 256 threads. I % 64 == 0.
extern "C" __global__ void moe4_dn_gemv(const float* __restrict__ act, const unsigned long long* __restrict__ eptr,
                                        const int* __restrict__ ustart, const int* __restrict__ psrc,
                                        const float* __restrict__ wts, float* __restrict__ ypick, int D, int I, int offq,
                                        int offs, int offm) {
    int lane = threadIdx.x & 31, o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), u = blockIdx.y;
    if (o >= D) return;
    const unsigned char* base = (const unsigned char*)eptr[u];
    const uint4* q = (const uint4*)(base + offq + (size_t)o * (I / 2));
    const unsigned short* sc = (const unsigned short*)(base + offs) + (size_t)o * (I / 64);
    const unsigned short* mn = (const unsigned short*)(base + offm) + (size_t)o * (I / 64);
    for (int p = ustart[u]; p < ustart[u + 1]; p++) {
        const float4* a = (const float4*)(act + (size_t)p * I);
        float acc = 0.f;
        for (int c = lane; c < I / 32; c += 32) {  // 32 codes (half a group) per uint4
            uint4 w = __ldg(q + c);
            const float4* av = a + c * 8;
            float4 a0 = av[0], a1 = av[1], a2 = av[2], a3 = av[3], a4 = av[4], a5 = av[5], a6 = av[6], a7 = av[7];
            float d = dot8_u4(w.x, a0, a1) + dot8_u4(w.y, a2, a3) + dot8_u4(w.z, a4, a5) + dot8_u4(w.w, a6, a7);
            acc += h2f(sc[c >> 1]) * d + h2f(mn[c >> 1]) * (sum8(a0, a1) + sum8(a2, a3) + sum8(a4, a5) + sum8(a6, a7));
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            int src = psrc[p];
            ypick[(size_t)src * D + o] = acc * wts[src];
        }
    }
}

// --- PLE: per-layer n-gram embeddings ---------------------------------------------------------

// e[i] = e4m3(rows[i]) * scale (the gathered table rows, one scale for the whole table).
extern "C" __global__ void ple_embed(const unsigned char* __restrict__ rows, const bf16_t* __restrict__ scale,
                                     float* __restrict__ e, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) e[i] = e4m3(rows[i]) * bf2f(scale[0]);
}

// Per lane l: g = (k_l . q_l) / sqrt(D), g = sign(g) sqrt(max(|g|, 1e-6)); gv[t, l, :] = sigmoid(g) v[t].
// k, q [T, HC*D] (normalised), v [T, D]. Grid (T, HC), block 256.
extern "C" __global__ void ple_gate(const float* __restrict__ k, const float* __restrict__ q, const float* __restrict__ v,
                                    float* __restrict__ gv, int D, int HC) {
    __shared__ float red[32];
    int t = blockIdx.x, l = blockIdx.y;
    size_t r = ((size_t)t * HC + l) * D;
    float a = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) a += k[r + i] * q[r + i];
    float g = block_sum(a, red) / sqrtf((float)D);
    g = g > 0.f ? sqrtf(fmaxf(g, 1e-6f)) : (g < 0.f ? -sqrtf(fmaxf(-g, 1e-6f)) : 0.f);
    float s = sigmoidf_(g);
    for (int i = threadIdx.x; i < D; i += blockDim.x) gv[r + i] = s * v[(size_t)t * D + i];
}

// Dilated (3) depthwise causal conv, kernel 4, over the normalised gvn, then SiLU; the lanes get
// gv + conv. w [C, 4] bf16 (w[3] takes the newest input); state [C, 9] fp32 (oldest first), the
// last 9 inputs. One thread per channel.
extern "C" __global__ void ple_conv(const float* __restrict__ gvn, const float* __restrict__ gv, float* __restrict__ x,
                                    const bf16_t* __restrict__ w, float* __restrict__ state, int T, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s[9];
#pragma unroll
    for (int j = 0; j < 9; j++) s[j] = state[(size_t)c * 9 + j];
    float w0 = bf2f(w[c * 4]), w1 = bf2f(w[c * 4 + 1]), w2 = bf2f(w[c * 4 + 2]), w3 = bf2f(w[c * 4 + 3]);
    for (int t = 0; t < T; t++) {
        size_t o = (size_t)t * C + c;
        float cur = gvn[o];
        x[o] += gv[o] + silu(w0 * s[0] + w1 * s[3] + w2 * s[6] + w3 * cur);
#pragma unroll
        for (int j = 0; j < 8; j++) s[j] = s[j + 1];
        s[8] = cur;
    }
#pragma unroll
    for (int j = 0; j < 9; j++) state[(size_t)c * 9 + j] = s[j];
}

// --- QSA indexer ---------------------------------------------------------------------------------

// Rotate-half rope of the first RD of HD dims at position pos, in a block of HD threads. buf: HD floats.
__device__ __forceinline__ float rope_dim(float nv, float* buf, int i, int RD, int pos, float theta) {
    buf[i] = nv;
    __syncthreads();
    float out = nv;
    if (i < RD) {
        int half = RD / 2, j = i < half ? i : i - half;
        float inv = 1.0f / powf(theta, (float)(2 * j) / (float)RD);
        float ang = (float)pos * inv;
        float c = cosf(ang), s = sinf(ang);
        out = i < half ? nv * c - buf[i + half] * s : nv * c + buf[i - half] * s;
    }
    return out;
}

// qk [T, (H+1)*HD]: H query heads then one key. Queries: zero-centred RMSNorm (qw) and rope at
// pos0 + t into qi [T, H, HD]; the raw key into kraw [cap, HD] at pos0 + t. Grid (T, H+1), block HD.
extern "C" __global__ void idx_prep(const float* __restrict__ qk, const bf16_t* __restrict__ qw, float* __restrict__ qi,
                                    float* __restrict__ kraw, int H, int HD, int RD, int pos0, float eps, float theta) {
    __shared__ float red[32];
    __shared__ float buf[256];
    int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    float xv = qk[((size_t)t * (H + 1) + h) * HD + i];
    if (h == H) {
        kraw[(size_t)(pos0 + t) * HD + i] = xv;
        return;
    }
    float ss = block_sum(xv * xv, red);
    float nv = (1.f + bf2f(qw[i])) * (xv * rsqrtf(ss / HD + eps));
    qi[((size_t)t * H + h) * HD + i] = rope_dim(nv, buf, i, RD, pos0 + t, theta);
}

// Block keys for blocks b0 .. b0 + gridDim.x - 1: the mean of R raw keys, zero-centred RMSNorm (kw),
// rope at the block's first position. bk [cap / R, HD]. Block HD.
extern "C" __global__ void idx_blocks(const float* __restrict__ kraw, const bf16_t* __restrict__ kw, float* __restrict__ bk,
                                      int b0, int R, int HD, int RD, float eps, float theta) {
    __shared__ float red[32];
    __shared__ float buf[256];
    int b = b0 + blockIdx.x, i = threadIdx.x;
    float m = 0.f;
    for (int j = 0; j < R; j++) m += kraw[((size_t)b * R + j) * HD + i];
    m /= R;
    float ss = block_sum(m * m, red);
    float nv = (1.f + bf2f(kw[i])) * (m * rsqrtf(ss / HD + eps));
    bk[(size_t)b * HD + i] = rope_dim(nv, buf, i, RD, b * R, theta);
}

// sc[t, b] = sum over heads of relu(qi[t, h] . bk[b]) / sqrt(HD), for the blocks complete at the
// query's position ((pos0 + t + 1) / R of them). One warp per block; grid (T, ceil(nb / 8)), 256 threads.
extern "C" __global__ void idx_scores(const float* __restrict__ qi, const float* __restrict__ bk, float* __restrict__ sc,
                                      int H, int HD, int pos0, int R, int nbmax) {
    int t = blockIdx.x, lane = threadIdx.x & 31;
    int b = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (b >= (pos0 + t + 1) / R) return;
    const float* kb = bk + (size_t)b * HD;
    float s = 0.f;
    for (int h = 0; h < H; h++) {
        const float* q = qi + ((size_t)t * H + h) * HD;
        float d = 0.f;
        for (int i = lane; i < HD; i += 32) d += q[i] * kb[i];
        d = warp_sum(d);
        s += d > 0.f ? d : 0.f;
    }
    if (lane == 0) sc[(size_t)t * nbmax + b] = s / sqrtf((float)HD);
}

// Each query's attended blocks: all n complete blocks when n <= top, else the top highest scores
// (ties to the lower block), found by a radix select on the (non-negative) scores' bits, listed in
// block order. blist [T, top], nsel [T]. One block of 256 threads per query.
extern "C" __global__ void idx_select(const float* __restrict__ sc, int* __restrict__ blist, int* __restrict__ nsel,
                                      int top, int R, int pos0, int nbmax) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned int s_prefix, s_mask;
    __shared__ int s_need, gt_at[257], eq_at[257];
    int t = blockIdx.x, tid = threadIdx.x;
    int n = (pos0 + t + 1) / R;
    const float* s = sc + (size_t)t * nbmax;
    int* out = blist + (size_t)t * top;
    if (n <= top) {
        for (int b = tid; b < n; b += blockDim.x) out[b] = b;
        if (tid == 0) nsel[t] = n;
        return;
    }
    if (tid == 0) {
        s_prefix = 0;
        s_mask = 0;
        s_need = top;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[tid] = 0;
        __syncthreads();
        unsigned int pf = s_prefix, mk = s_mask;
        for (int b = tid; b < n; b += blockDim.x) {
            unsigned int u = __float_as_uint(s[b]);
            if ((u & mk) == pf) atomicAdd(&hist[(u >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            int need = s_need, bin = 255;
            for (; bin > 0; bin--) {
                if ((int)hist[bin] >= need) break;
                need -= hist[bin];
            }
            s_prefix = pf | ((unsigned int)bin << shift);
            s_mask = mk | (255u << shift);
            s_need = need;
        }
        __syncthreads();
    }
    unsigned int thr = s_prefix;
    int take_eq = s_need;  // how many scores equal to thr are taken (the rest of top are above it)
    int per = (n + blockDim.x - 1) / blockDim.x, lo = min(n, tid * per), hi = min(n, lo + per);
    int gt = 0, eq = 0;
    for (int b = lo; b < hi; b++) {
        unsigned int u = __float_as_uint(s[b]);
        gt += u > thr;
        eq += u == thr;
    }
    gt_at[tid + 1] = gt;
    eq_at[tid + 1] = eq;
    __syncthreads();
    if (tid == 0) {
        gt_at[0] = eq_at[0] = 0;
        for (int i = 1; i <= (int)blockDim.x; i++) {
            gt_at[i] += gt_at[i - 1];
            eq_at[i] += eq_at[i - 1];
        }
    }
    __syncthreads();
    // blocks above thr, and the first take_eq equal to it, in block order: position = how many
    // chosen blocks come before
    int g = gt_at[tid], e = eq_at[tid];
    for (int b = lo; b < hi; b++) {
        unsigned int u = __float_as_uint(s[b]);
        if (u > thr) {
            out[g + min(e, take_eq)] = b;
            g++;
        } else if (u == thr) {
            if (e < take_eq) out[g + e] = b;
            e++;
        }
    }
    if (tid == 0) nsel[t] = top;
}

// attention_gated over a list of keys: the query at pos0 + t sees the R keys of each of its nsel[t]
// listed blocks and the tail of its incomplete block (positions from complete * R to its own).
extern "C" __global__ void attention_gated_list(const float* __restrict__ q, const bf16_t* __restrict__ kc,
                                                const bf16_t* __restrict__ vc, const float* __restrict__ qg,
                                                float* __restrict__ out, int H, int KV, int pos0, int cap, float scale,
                                                const int* __restrict__ blist, const int* __restrict__ nsel, int top,
                                                int R) {
    const int HD = 256, NW = 8;
    __shared__ float sm[NW], ssum[NW];
    __shared__ float sacc[NW][HD];
    int t = blockIdx.x, h = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int kvh = h / (H / KV);
    const float4* qp = (const float4*)(q + ((size_t)t * H + h) * HD) + lane * 2;
    float4 q0 = qp[0], q1 = qp[1];
    int p = pos0 + t, complete = (p + 1) / R, ns = nsel[t];
    const int* bl = blist + (size_t)t * top;
    int L = ns * R + (p + 1 - complete * R);
    const bf16_t* kb = kc + (size_t)kvh * cap * HD;
    const bf16_t* vb = vc + (size_t)kvh * cap * HD;
    float m = neg_inf(), s = 0.f, a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int j = warp; j < L; j += NW) {
        int l = j < ns * R ? bl[j / R] * R + j % R : complete * R + (j - ns * R);
        uint4 kp = ((const uint4*)(kb + (size_t)l * HD))[lane];
        float d = q0.x * bf_lo(kp.x) + q0.y * bf_hi(kp.x) + q0.z * bf_lo(kp.y) + q0.w * bf_hi(kp.y) +
                  q1.x * bf_lo(kp.z) + q1.y * bf_hi(kp.z) + q1.z * bf_lo(kp.w) + q1.w * bf_hi(kp.w);
        d = warp_sum(d) * scale;
        float mn = fmaxf(m, d);
        float corr = expf(m - mn), pr = expf(d - mn);
        uint4 vp = ((const uint4*)(vb + (size_t)l * HD))[lane];
        s = s * corr + pr;
        a[0] = a[0] * corr + pr * bf_lo(vp.x);
        a[1] = a[1] * corr + pr * bf_hi(vp.x);
        a[2] = a[2] * corr + pr * bf_lo(vp.y);
        a[3] = a[3] * corr + pr * bf_hi(vp.y);
        a[4] = a[4] * corr + pr * bf_lo(vp.z);
        a[5] = a[5] * corr + pr * bf_hi(vp.z);
        a[6] = a[6] * corr + pr * bf_lo(vp.w);
        a[7] = a[7] * corr + pr * bf_hi(vp.w);
        m = mn;
    }
    if (lane == 0) {
        sm[warp] = m;
        ssum[warp] = s;
    }
#pragma unroll
    for (int j = 0; j < 8; j++) sacc[warp][lane * 8 + j] = a[j];
    __syncthreads();
    int i = threadIdx.x;
    float M = neg_inf();
    for (int w = 0; w < NW; w++) M = fmaxf(M, sm[w]);
    float S = 0.f, A = 0.f;
    for (int w = 0; w < NW; w++) {
        if (sm[w] == neg_inf()) continue;
        float f = expf(sm[w] - M);
        S += ssum[w] * f;
        A += sacc[w][i] * f;
    }
    float gate = qg[((size_t)t * H + h) * 2 * HD + HD + i];
    out[((size_t)t * H + h) * HD + i] = (A / S) * sigmoidf_(gate);
}
