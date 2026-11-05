// Kimi-K2.6 deep brain (aegist_rs/src/kimi.rs): int4 experts as published, and MLA attention.
//
// Experts, format "q4g32s" (aegist/fetch_k2.py), one record: [gate codes][gate scales][up codes][up
// scales][down codes][down scales]. A matrix [rows, K]: codes are K/8 int32 words per row, value i of
// a word in bits 4i..4i+3 stored as v + 8; one bf16 scale per 32 values; weight = scale * (nibble - 8).
// So a group's dot product is scale * (nibbles . x - 8 * sum(x)).
//
// MLA: the cache holds, per position, the normalised latent c (512) and the roped shared key part
// kpe (64). Attention runs in the latent space: q_lat = W_uk^T q_nope per head, score =
// scale * (q_lat . c + q_pe . kpe), o_lat = sum p c, out = W_uv o_lat (W_uk / W_uv: kv_b_proj rows).

__device__ __forceinline__ float k2_dot8(unsigned int w, float4 a, float4 b) {
    return a.x * (float)(w & 15u) + a.y * (float)((w >> 4) & 15u) + a.z * (float)((w >> 8) & 15u) +
           a.w * (float)((w >> 12) & 15u) + b.x * (float)((w >> 16) & 15u) + b.y * (float)((w >> 20) & 15u) +
           b.z * (float)((w >> 24) & 15u) + b.w * (float)(w >> 28);
}

// scale * (nibbles . x - 8 * sum(x)) for one group of 32; x 16-byte aligned.
__device__ __forceinline__ float k2_group_dot(uint4 q, unsigned short sc, const float* x) {
    const float4* xv = (const float4*)x;
    float d = k2_dot8(q.x, xv[0], xv[1]) + k2_dot8(q.y, xv[2], xv[3]) + k2_dot8(q.z, xv[4], xv[5]) +
              k2_dot8(q.w, xv[6], xv[7]);
    float sx = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) sx += xv[j].x + xv[j].y + xv[j].z + xv[j].w;
    return bf2f(sc) * (d - 8.f * sx);
}

// SwiGLU input side for few picks per expert: act[p, r] = silu(gate_r . x) * (up_r . x).
// Offsets (bytes, inside a record): gate codes 0, gate scales gs, up codes uq, up scales us.
// One warp per r; grid (I/8, U), 256 threads.
extern "C" __global__ void k2q4_gu_gemv(const float* __restrict__ h, const unsigned long long* __restrict__ eptr,
                                        const int* __restrict__ ustart, const int* __restrict__ ptok,
                                        float* __restrict__ act, int D, int I, int gs, int uq, int us) {
    int lane = threadIdx.x & 31, r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), u = blockIdx.y;
    if (r >= I) return;
    const unsigned char* base = (const unsigned char*)eptr[u];
    int G = D / 32;
    const uint4* qg = (const uint4*)base + (size_t)r * G;
    const uint4* qu = (const uint4*)(base + uq) + (size_t)r * G;
    const unsigned short* sg = (const unsigned short*)(base + gs) + (size_t)r * G;
    const unsigned short* su = (const unsigned short*)(base + us) + (size_t)r * G;
    for (int p = ustart[u]; p < ustart[u + 1]; p++) {
        const float* x = h + (size_t)ptok[p] * D;
        float ag = 0.f, au = 0.f;
        for (int g = lane; g < G; g += 32) {
            ag += k2_group_dot(__ldg(qg + g), sg[g], x + g * 32);
            au += k2_group_dot(__ldg(qu + g), su[g], x + g * 32);
        }
        ag = warp_sum(ag);
        au = warp_sum(au);
        if (lane == 0) act[(size_t)p * I + r] = silu(ag) * au;
    }
}

// SwiGLU output side for few picks per expert: ypick[psrc[p], o] = wts[psrc[p]] * (down_o . act[p]).
// Down codes at dq, scales at ds. One warp per output row; grid (D/8, U), 256 threads.
extern "C" __global__ void k2q4_dn_gemv(const float* __restrict__ act, const unsigned long long* __restrict__ eptr,
                                         const int* __restrict__ ustart, const int* __restrict__ psrc,
                                         const float* __restrict__ wts, float* __restrict__ ypick, int D, int I, int dq,
                                         int ds) {
    int lane = threadIdx.x & 31, o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), u = blockIdx.y;
    if (o >= D) return;
    const unsigned char* base = (const unsigned char*)eptr[u];
    int G = I / 32;
    const uint4* q = (const uint4*)(base + dq) + (size_t)o * G;
    const unsigned short* s = (const unsigned short*)(base + ds) + (size_t)o * G;
    for (int p = ustart[u]; p < ustart[u + 1]; p++) {
        const float* a = act + (size_t)p * I;
        float v = 0.f;
        for (int g = lane; g < G; g += 32) v += k2_group_dot(__ldg(q + g), s[g], a + g * 32);
        v = warp_sum(v);
        if (lane == 0) {
            int src = psrc[p];
            ypick[(size_t)src * D + o] = v * wts[src];
        }
    }
}

// Dequantise rows [row0, row0 + n) x columns [64 t, 64 t + 64) of an int4 matrix (G groups of 32 per
// row) into Ws[n][65].
__device__ __forceinline__ void k2_load_tile(float (*Ws)[65], const unsigned int* q, const unsigned short* sc,
                                             int row0, int n, int G, int t) {
    for (int i = threadIdx.x; i < n * 8; i += blockDim.x) {
        int r = i >> 3, j = i & 7;  // 8 words of 8 values per row and tile
        int row = row0 + r;
        unsigned int w = q[(size_t)row * G * 4 + t * 8 + j];
        float s = bf2f(sc[(size_t)row * G + t * 2 + (j >> 2)]);
#pragma unroll
        for (int b = 0; b < 8; b++) Ws[r][j * 8 + b] = ((float)((w >> (4 * b)) & 15u) - 8.f) * s;
    }
}

// SwiGLU input side for many picks per expert (prefill). Grid (I/32, U), 256 threads.
extern "C" __global__ void k2q4_gu_gemm(const float* __restrict__ h, const unsigned long long* __restrict__ eptr,
                                        const int* __restrict__ ustart, const int* __restrict__ ptok,
                                        float* __restrict__ act, int D, int I, int gs, int uq, int us) {
    __shared__ float Xs[32][65];
    __shared__ float Ws[64][65];
    int u = blockIdx.y, r0 = blockIdx.x * 32, tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const unsigned char* base = (const unsigned char*)eptr[u];
    const unsigned short* sg = (const unsigned short*)(base + gs);
    const unsigned short* su = (const unsigned short*)(base + us);
    int G = D / 32, T = D / 64, p_end = ustart[u + 1];
    for (int p0 = ustart[u]; p0 < p_end; p0 += 32) {
        float ag[2][2] = {{0.f, 0.f}, {0.f, 0.f}}, au[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
        for (int t = 0; t < T; t++) {
            for (int i = tid; i < 32 * 64; i += blockDim.x) {
                int pr = i >> 6, c = i & 63, p = p0 + pr;
                Xs[pr][c] = p < p_end ? h[(size_t)ptok[p] * D + t * 64 + c] : 0.f;
            }
            k2_load_tile(Ws, (const unsigned int*)base, sg, r0, 32, G, t);
            k2_load_tile(Ws + 32, (const unsigned int*)(base + uq), su, r0, 32, G, t);
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

// SwiGLU output side for many picks per expert (prefill). Grid (D/64, U), 256 threads.
extern "C" __global__ void k2q4_dn_gemm(const float* __restrict__ act, const unsigned long long* __restrict__ eptr,
                                         const int* __restrict__ ustart, const int* __restrict__ psrc,
                                         const float* __restrict__ wts, float* __restrict__ ypick, int D, int I, int dq,
                                         int ds) {
    __shared__ float Xs[32][65];
    __shared__ float Ws[64][65];
    int u = blockIdx.y, o0 = blockIdx.x * 64, tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const unsigned char* base = (const unsigned char*)eptr[u];
    const unsigned short* sc = (const unsigned short*)(base + ds);
    int G = I / 32, T = I / 64, p_end = ustart[u + 1];
    for (int p0 = ustart[u]; p0 < p_end; p0 += 32) {
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int t = 0; t < T; t++) {
            for (int i = tid; i < 32 * 64; i += blockDim.x) {
                int pr = i >> 6, c = i & 63, p = p0 + pr;
                Xs[pr][c] = p < p_end ? act[(size_t)p * I + t * 64 + c] : 0.f;
            }
            k2_load_tile(Ws, (const unsigned int*)(base + dq), sc, o0, 64, G, t);
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

// Router (sigmoid scores, "noaux_tc" with one group): experts chosen by score + bias, weighted by the
// plain score, normalised over the chosen ones, times `scale`. One block (256 threads) per token.
extern "C" __global__ void k2_router(const float* __restrict__ h, const bf16_t* __restrict__ W,
                                     const float* __restrict__ bias, int* __restrict__ idx, float* __restrict__ wts,
                                     int D, int E, int K, float scale) {
    __shared__ float s[1024], c[1024];
    int t = blockIdx.x, lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const float* x = h + (size_t)t * D;
    for (int e = wid; e < E; e += nw) {
        const bf16_t* row = W + (size_t)e * D;
        float d = 0.f;
        for (int i = lane; i < D; i += 32) d += x[i] * bf2f(row[i]);
        d = warp_sum(d);
        if (lane == 0) {
            float sg = 1.f / (1.f + expf(-d));
            s[e] = sg;
            c[e] = sg + bias[e];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int k = 0; k < K; k++) {
            int bi = 0;
            float best = neg_inf();
            for (int e = 0; e < E; e++)
                if (c[e] > best) {
                    best = c[e];
                    bi = e;
                }
            c[bi] = neg_inf();
            idx[t * K + k] = bi;
            wts[t * K + k] = s[bi];
            sum += s[bi];
        }
        float m = scale / (sum + 1e-20f);
        for (int k = 0; k < K; k++) wts[t * K + k] *= m;
    }
}

// Rotate the 64 rope values at v (interleaved pairs) for position pos. Threads 0..31 of the block.
__device__ __forceinline__ void k2_rope_pairs(float* v, const float* __restrict__ inv_freq, int pos) {
    int i = threadIdx.x;
    if (i < 32) {
        float sn, cs;
        sincosf((float)pos * inv_freq[i], &sn, &cs);
        float a = v[2 * i], b = v[2 * i + 1];
        v[2 * i] = a * cs - b * sn;
        v[2 * i + 1] = b * cs + a * sn;
    }
}

// kv [t, 576] (kv_a_proj output) -> cache: ckv[pos] = rmsnorm(kv[:512]) * nw, kpe[pos] = rope(kv[512:]).
// One block (256 threads) per token.
extern "C" __global__ void mla_kv(float* __restrict__ kv, const bf16_t* __restrict__ nw,
                                  const float* __restrict__ inv_freq, float* __restrict__ ckv, float* __restrict__ kpe,
                                  int pos0, float eps) {
    __shared__ float red[32];
    int t = blockIdx.x, pos = pos0 + t;
    float* r = kv + (size_t)t * 576;
    float ss = 0.f;
    for (int i = threadIdx.x; i < 512; i += blockDim.x) ss += r[i] * r[i];
    ss = block_sum(ss, red);
    float inv = rsqrtf(ss / 512.f + eps);
    float* cr = ckv + (size_t)pos * 512;
    for (int i = threadIdx.x; i < 512; i += blockDim.x) cr[i] = bf2f(nw[i]) * (r[i] * inv);
    k2_rope_pairs(r + 512, inv_freq, pos);
    __syncthreads();
    for (int i = threadIdx.x; i < 64; i += blockDim.x) kpe[(size_t)pos * 64 + i] = r[512 + i];
}

// q [t, H, 192] (nope 128 then rope 64): rope q_pe in place; qlat[t, h] = W_uk[h]^T q_nope[t, h]
// with W_uk[h] = kv_b rows h*256 .. h*256+127 (each 512 wide). Grid (t, H), 256 threads.
extern "C" __global__ void mla_q(float* __restrict__ q, const bf16_t* __restrict__ kvb,
                                 const float* __restrict__ inv_freq, float* __restrict__ qlat, int H, int pos0) {
    __shared__ float qn[128];
    int t = blockIdx.x, h = blockIdx.y;
    float* qh = q + ((size_t)t * H + h) * 192;
    for (int i = threadIdx.x; i < 128; i += blockDim.x) qn[i] = qh[i];
    k2_rope_pairs(qh + 128, inv_freq, pos0 + t);
    __syncthreads();
    const bf16_t* w = kvb + (size_t)h * 256 * 512;
    float* out = qlat + ((size_t)t * H + h) * 512;
    for (int c = threadIdx.x; c < 512; c += blockDim.x) {
        float acc = 0.f;
#pragma unroll 8
        for (int i = 0; i < 128; i++) acc += qn[i] * bf2f(w[(size_t)i * 512 + c]);
        out[c] = acc;
    }
}

// Causal attention in the latent space for queries at positions pos0 + t (keys 0..pos0+t), online
// softmax over tiles of 256 keys. Grid (t, H), 256 threads: warps score keys, every thread owns two
// of the 512 output values.
extern "C" __global__ void mla_attn(const float* __restrict__ qlat, const float* __restrict__ q,
                                    const float* __restrict__ ckv, const float* __restrict__ kpe,
                                    float* __restrict__ olat, int H, int pos0, float scale) {
    __shared__ float qs[576];
    __shared__ float sc[256];
    __shared__ float red[32];
    int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, nw = blockDim.x >> 5;
    const float* ql = qlat + ((size_t)t * H + h) * 512;
    const float* qp = q + ((size_t)t * H + h) * 192 + 128;
    for (int i = tid; i < 576; i += blockDim.x) qs[i] = i < 512 ? ql[i] : qp[i - 512];
    __syncthreads();
    int n = pos0 + t + 1;
    float m = neg_inf(), l = 0.f, o0 = 0.f, o1 = 0.f;
    for (int j0 = 0; j0 < n; j0 += 256) {
        int cnt = min(256, n - j0);
        for (int k = wid; k < cnt; k += nw) {
            const float* cr = ckv + (size_t)(j0 + k) * 512;
            const float* pr = kpe + (size_t)(j0 + k) * 64;
            float d = 0.f;
            for (int i = lane; i < 512; i += 32) d += qs[i] * cr[i];
            for (int i = lane; i < 64; i += 32) d += qs[512 + i] * pr[i];
            d = warp_sum(d);
            if (lane == 0) sc[k] = d * scale;
        }
        __syncthreads();
        float mx = neg_inf();
        for (int k = tid; k < cnt; k += blockDim.x) mx = fmaxf(mx, sc[k]);
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        if (lane == 0) red[wid] = mx;
        __syncthreads();
        mx = neg_inf();
        for (int w = 0; w < nw; w++) mx = fmaxf(mx, red[w]);
        float mnew = fmaxf(m, mx);
        float corr = __expf(m - mnew);  // 0 on the first tile
        __syncthreads();
        for (int k = tid; k < cnt; k += blockDim.x) sc[k] = __expf(sc[k] - mnew);
        __syncthreads();
        float ps = 0.f;
        for (int k = tid; k < cnt; k += blockDim.x) ps += sc[k];
        ps = block_sum(ps, red);
        l = l * corr + ps;
        o0 *= corr;
        o1 *= corr;
        for (int k = 0; k < cnt; k++) {
            float p = sc[k];
            const float* cr = ckv + (size_t)(j0 + k) * 512;
            o0 += p * cr[tid];
            o1 += p * cr[tid + 256];
        }
        m = mnew;
        __syncthreads();
    }
    float* out = olat + ((size_t)t * H + h) * 512;
    out[tid] = o0 / l;
    out[tid + 256] = o1 / l;
}

// att[t, h*128 + i] = W_uv[h, i] . olat[t, h] with W_uv[h, i] = kv_b row h*256 + 128 + i.
// Grid (t, H), 256 threads, one warp per output value.
extern "C" __global__ void mla_out(const float* __restrict__ olat, const bf16_t* __restrict__ kvb,
                                   float* __restrict__ att, int H) {
    __shared__ float os[512];
    int t = blockIdx.x, h = blockIdx.y, lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const float* src = olat + ((size_t)t * H + h) * 512;
    for (int i = threadIdx.x; i < 512; i += blockDim.x) os[i] = src[i];
    __syncthreads();
    for (int i = wid; i < 128; i += nw) {
        const bf16_t* w = kvb + ((size_t)h * 256 + 128 + i) * 512;
        float d = 0.f;
        for (int c = lane; c < 512; c += 32) d += os[c] * bf2f(w[c]);
        d = warp_sum(d);
        if (lane == 0) att[((size_t)t * H + h) * 128 + i] = d;
    }
}
