// Hybrid-MoE kernels (arch "hybrid-moe": gated DeltaNet + gated attention + 4-bit experts).
// Included at the end of aegist.cu, so it shares its helpers (bf16, block_sum, silu, ...).
// Same conventions: fp32 activations, int8 / bf16 / 4-bit weights widened on the fly.

__device__ __forceinline__ float h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ float sigmoidf_(float a) { return 1.f / (1.f + expf(-a)); }

// Max over the block (same contract as block_sum).
__device__ float block_max(float v, float* red) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (lane == 0) red[w] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : neg_inf();
    if (w == 0)
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (threadIdx.x == 0) red[0] = v;
    __syncthreads();
    float r = red[0];
    __syncthreads();
    return r;
}

// Largest v over the block, ties to the lower index. Every thread gets (value, index).
__device__ void block_argmax(float v, int i, float* rv, int* ri, float* ov, int* oi) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        float v2 = __shfl_xor_sync(0xffffffffu, v, o);
        int i2 = __shfl_xor_sync(0xffffffffu, i, o);
        if (v2 > v || (v2 == v && i2 < i)) {
            v = v2;
            i = i2;
        }
    }
    if (lane == 0) {
        rv[w] = v;
        ri[w] = i;
    }
    __syncthreads();
    if (w == 0) {
        bool ok = threadIdx.x < (blockDim.x >> 5);
        v = ok ? rv[threadIdx.x] : neg_inf();
        i = ok ? ri[threadIdx.x] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) {
            float v2 = __shfl_xor_sync(0xffffffffu, v, o);
            int i2 = __shfl_xor_sync(0xffffffffu, i, o);
            if (v2 > v || (v2 == v && i2 < i)) {
                v = v2;
                i = i2;
            }
        }
        if (threadIdx.x == 0) {
            rv[0] = v;
            ri[0] = i;
        }
    }
    __syncthreads();
    *ov = rv[0];
    *oi = ri[0];
    __syncthreads();
}

// Zero-centred RMSNorm: out = (1 + w) * x / rms(x), one block per row.
extern "C" __global__ void znorm(const float* __restrict__ x, const bf16_t* __restrict__ w, float* __restrict__ out,
                                 int D, float eps) {
    __shared__ float red[32];
    const float* xr = x + (size_t)blockIdx.x * D;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, red);
    float inv = rsqrtf(ss / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[(size_t)blockIdx.x * D + i] = (1.f + bf2f(w[i])) * (xr[i] * inv);
}

// --- gated attention ---------------------------------------------------------------------

// Per-head zero-centred RMSNorm of the query (the first HD of each head's 2*HD in qg) and of the
// key, then rotary embedding on the first RD dims (rotate-half form); the rest pass through.
// One block of HD threads (HD <= 256) per (token, head). Queries go to qn [T, H, HD]; keys and
// values into this layer's cache at position pos0 + token. Cache layout: [KV, cap, HD] bf16.
extern "C" __global__ void attn_prep(const float* __restrict__ qg, const float* __restrict__ k, const float* __restrict__ v,
                                     const bf16_t* __restrict__ qw, const bf16_t* __restrict__ kw, float* __restrict__ qn,
                                     bf16_t* __restrict__ kc, bf16_t* __restrict__ vc, int H, int KV, int HD, int RD,
                                     int pos0, int cap, float eps, float theta) {
    __shared__ float red[32];
    __shared__ float buf[256];
    int t = blockIdx.x, hh = blockIdx.y, i = threadIdx.x;
    bool isq = hh < H;
    int h = isq ? hh : hh - H;
    const float* src = isq ? qg + ((size_t)t * H + h) * 2 * HD : k + ((size_t)t * KV + h) * HD;
    float xv = src[i];
    float ss = block_sum(xv * xv, red);
    float nv = (1.f + bf2f(isq ? qw[i] : kw[i])) * (xv * rsqrtf(ss / HD + eps));
    buf[i] = nv;
    __syncthreads();
    float out = nv;
    if (i < RD) {
        int half = RD / 2, j = i < half ? i : i - half;
        float inv = 1.0f / powf(theta, (float)(2 * j) / (float)RD);
        float ang = (float)(pos0 + t) * inv;
        float c = cosf(ang), s = sinf(ang);
        out = i < half ? nv * c - buf[i + half] * s : nv * c + buf[i - half] * s;
    }
    if (isq) {
        qn[((size_t)t * H + h) * HD + i] = out;
    } else {
        size_t o = ((size_t)h * cap + pos0 + t) * HD + i;
        kc[o] = f2bf(out);
        vc[o] = f2bf(v[((size_t)t * KV + h) * HD + i]);
    }
}

// Causal GQA attention for head_dim 256 with an online softmax, then the output gate:
// out = attention * sigmoid(gate), gate = the second HD of each head in qg.
// One block of 8 warps (256 threads) per (token, query head); each lane owns 8 of the 256 dims.
extern "C" __global__ void attention_gated(const float* __restrict__ q, const bf16_t* __restrict__ kc,
                                           const bf16_t* __restrict__ vc, const float* __restrict__ qg,
                                           float* __restrict__ out, int H, int KV, int pos0, int cap, float scale) {
    const int HD = 256, NW = 8;
    __shared__ float sm[NW], ssum[NW];
    __shared__ float sacc[NW][HD];
    int t = blockIdx.x, h = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int kvh = h / (H / KV);
    const float4* qp = (const float4*)(q + ((size_t)t * H + h) * HD) + lane * 2;
    float4 q0 = qp[0], q1 = qp[1];
    int L = pos0 + t + 1;
    const bf16_t* kb = kc + (size_t)kvh * cap * HD;
    const bf16_t* vb = vc + (size_t)kvh * cap * HD;
    float m = neg_inf(), s = 0.f, a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int l = warp; l < L; l += NW) {
        uint4 kp = ((const uint4*)(kb + (size_t)l * HD))[lane];
        float d = q0.x * bf_lo(kp.x) + q0.y * bf_hi(kp.x) + q0.z * bf_lo(kp.y) + q0.w * bf_hi(kp.y) +
                  q1.x * bf_lo(kp.z) + q1.y * bf_hi(kp.z) + q1.z * bf_lo(kp.w) + q1.w * bf_hi(kp.w);
        d = warp_sum(d) * scale;
        float mn = fmaxf(m, d);
        float corr = expf(m - mn), p = expf(d - mn);
        uint4 vp = ((const uint4*)(vb + (size_t)l * HD))[lane];
        s = s * corr + p;
        a[0] = a[0] * corr + p * bf_lo(vp.x);
        a[1] = a[1] * corr + p * bf_hi(vp.x);
        a[2] = a[2] * corr + p * bf_lo(vp.y);
        a[3] = a[3] * corr + p * bf_hi(vp.y);
        a[4] = a[4] * corr + p * bf_lo(vp.z);
        a[5] = a[5] * corr + p * bf_hi(vp.z);
        a[6] = a[6] * corr + p * bf_lo(vp.w);
        a[7] = a[7] * corr + p * bf_hi(vp.w);
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
        if (sm[w] == neg_inf()) continue;  // this warp saw no keys
        float f = expf(sm[w] - M);
        S += ssum[w] * f;
        A += sacc[w][i] * f;
    }
    float gate = qg[((size_t)t * H + h) * 2 * HD + HD + i];
    out[((size_t)t * H + h) * HD + i] = (A / S) * sigmoidf_(gate);
}

// --- gated DeltaNet ------------------------------------------------------------------------

// beta = sigmoid(h . Wb[j]); g = -exp(A_log[j]) * softplus(h . Wa[j] + dt_bias[j]).
// One block (256 threads) per token. gb [T, 2*HV]: g in [0, HV), beta in [HV, 2*HV).
extern "C" __global__ void delta_gates(const float* __restrict__ h, const bf16_t* __restrict__ wb,
                                       const bf16_t* __restrict__ wa, const bf16_t* __restrict__ a_log,
                                       const bf16_t* __restrict__ dt_bias, float* __restrict__ gb, int D, int HV) {
    int t = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const float* hr = h + (size_t)t * D;
    for (int j = warp; j < 2 * HV; j += nw) {
        const bf16_t* w = j < HV ? wb + (size_t)j * D : wa + (size_t)(j - HV) * D;
        float acc = 0.f;
        for (int i = lane; i < D; i += 32) acc += hr[i] * bf2f(w[i]);
        acc = warp_sum(acc);
        if (lane == 0) {
            if (j < HV) {
                gb[(size_t)t * 2 * HV + HV + j] = sigmoidf_(acc);
            } else {
                int jj = j - HV;
                float x = acc + bf2f(dt_bias[jj]);
                float sp = x > 20.f ? x : log1pf(expf(x));
                gb[(size_t)t * 2 * HV + jj] = -expf(bf2f(a_log[jj])) * sp;
            }
        }
    }
}

// Depthwise causal conv (kernel 4) over T new inputs, continuing from the 3 previous inputs in
// state, then SiLU. x, y [T, C]; w [C, 4] bf16 (w[3] multiplies the newest input);
// state [C, 3] fp32, oldest first, updated to the last 3 inputs. One thread per channel.
extern "C" __global__ void conv_silu(const float* __restrict__ x, float* __restrict__ y, const bf16_t* __restrict__ w,
                                     float* __restrict__ state, int T, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s0 = state[c * 3], s1 = state[c * 3 + 1], s2 = state[c * 3 + 2];
    float w0 = bf2f(w[c * 4]), w1 = bf2f(w[c * 4 + 1]), w2 = bf2f(w[c * 4 + 2]), w3 = bf2f(w[c * 4 + 3]);
    for (int t = 0; t < T; t++) {
        float xv = x[(size_t)t * C + c];
        y[(size_t)t * C + c] = silu(w0 * s0 + w1 * s1 + w2 * s2 + w3 * xv);
        s0 = s1;
        s1 = s2;
        s2 = xv;
    }
    state[c * 3] = s0;
    state[c * 3 + 1] = s1;
    state[c * 3 + 2] = s2;
}

// L2-normalise the 2*nq query and key heads (dk each) at the start of every row of y [T, C]
// in place; queries are also scaled by qscale. Grid (T, 2*nq), block dk.
extern "C" __global__ void l2norm_heads(float* __restrict__ y, int C, int nq, int dk, float qscale, float eps) {
    __shared__ float red[32];
    int t = blockIdx.x, hh = blockIdx.y, i = threadIdx.x;
    float* p = y + (size_t)t * C + (size_t)hh * dk;
    float v = p[i];
    float ss = block_sum(v * v, red);
    p[i] = v * rsqrtf(ss + eps) * (hh < nq ? qscale : 1.f);
}

// The gated delta rule, token by token, with the state in registers for the whole sequence.
// One block per value head, 512 threads: thread (column j = tid / 4, part = tid % 4) owns rows
// [part*32, part*32+32) of column j of the head's state S [DK, DV]. Per token:
//   S *= exp(g);  kv = S^T k;  delta = (v - kv) * beta;  S += k delta^T;  o = S^T q.
// y [T, C] holds q (HK heads), k (HK heads), v (HV heads) per row; value head h reads key/query
// head h / (HV / HK). gb from delta_gates. state [HV, DK, DV] fp32, updated. o [T, HV, DV].
#define DN_DK 128
#define DN_DV 128
#define DN_R 32
#define DN_PAD (DN_R + 1)
extern "C" __global__ void __launch_bounds__(512) delta_scan(const float* __restrict__ y, const float* __restrict__ gb,
                                                            float* __restrict__ state, float* __restrict__ o, int T,
                                                            int C, int HK, int HV) {
    __shared__ float sk[2][4 * DN_PAD], sq[2][4 * DN_PAD], sv[2][DN_DV], sg[2][2];
    int h = blockIdx.x, tid = threadIdx.x, j = tid >> 2, part = tid & 3;
    int kh = h / (HV / HK);
    float S[DN_R];
    float* st = state + (size_t)h * DN_DK * DN_DV;
#pragma unroll
    for (int r = 0; r < DN_R; r++) S[r] = st[(size_t)(part * DN_R + r) * DN_DV + j];

    const size_t qoff = (size_t)kh * DN_DK, koff = (size_t)HK * DN_DK + (size_t)kh * DN_DK,
                 voff = 2 * (size_t)HK * DN_DK + (size_t)h * DN_DV;
#define DN_LOAD(t_, b_)                                                       \
    {                                                                         \
        const float* row = y + (size_t)(t_) * C;                              \
        if (tid < DN_DK) {                                                    \
            sk[b_][(tid / DN_R) * DN_PAD + tid % DN_R] = row[koff + tid];     \
        } else if (tid < 2 * DN_DK) {                                         \
            int i_ = tid - DN_DK;                                             \
            sq[b_][(i_ / DN_R) * DN_PAD + i_ % DN_R] = row[qoff + i_];        \
        } else if (tid < 2 * DN_DK + DN_DV) {                                 \
            sv[b_][tid - 2 * DN_DK] = row[voff + tid - 2 * DN_DK];            \
        } else if (tid == 2 * DN_DK + DN_DV) {                                \
            sg[b_][0] = gb[(size_t)(t_) * 2 * HV + h];                        \
            sg[b_][1] = gb[(size_t)(t_) * 2 * HV + HV + h];                   \
        }                                                                     \
    }
    DN_LOAD(0, 0);
    __syncthreads();
    for (int t = 0; t < T; t++) {
        int b = t & 1;
        if (t + 1 < T) DN_LOAD(t + 1, b ^ 1);
        float decay = expf(sg[b][0]), beta = sg[b][1];
        const float* kk = sk[b] + part * DN_PAD;
        const float* qq = sq[b] + part * DN_PAD;
        float kv = 0.f;
#pragma unroll
        for (int r = 0; r < DN_R; r++) kv += S[r] * kk[r];
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        kv += __shfl_xor_sync(0xffffffffu, kv, 2);
        float delta = (sv[b][j] - decay * kv) * beta;
        float ov = 0.f;
#pragma unroll
        for (int r = 0; r < DN_R; r++) {
            S[r] = decay * S[r] + kk[r] * delta;
            ov += S[r] * qq[r];
        }
        ov += __shfl_xor_sync(0xffffffffu, ov, 1);
        ov += __shfl_xor_sync(0xffffffffu, ov, 2);
        if (part == 0) o[(size_t)t * HV * DN_DV + (size_t)h * DN_DV + j] = ov;
        __syncthreads();
    }
#undef DN_LOAD
#pragma unroll
    for (int r = 0; r < DN_R; r++) st[(size_t)(part * DN_R + r) * DN_DV + j] = S[r];
}

// out = w * rmsnorm(o) * silu(z) per value head. Grid (T, HV), block DV.
extern "C" __global__ void gated_norm(const float* __restrict__ o, const float* __restrict__ z,
                                      const bf16_t* __restrict__ w, float* __restrict__ out, int HV, int DV, float eps) {
    __shared__ float red[32];
    size_t base = ((size_t)blockIdx.x * HV + blockIdx.y) * DV;
    int i = threadIdx.x;
    float v = o[base + i];
    float ss = block_sum(v * v, red);
    out[base + i] = bf2f(w[i]) * (v * rsqrtf(ss / DV + eps)) * silu(z[base + i]);
}

// --- mixture of experts ------------------------------------------------------------------------

// Router: logits h . W[e] (bf16 W [E, D], D % 8 == 0), softmax in fp32, top K (K <= 8),
// renormalised. One block (256 threads, E <= 256) per token. idx, wts [T, K].
extern "C" __global__ void router_topk(const float* __restrict__ h, const bf16_t* __restrict__ W, int* __restrict__ idx,
                                       float* __restrict__ wts, int D, int E, int K) {
    __shared__ float lg[256], red[32], rv[32], sv[8];
    __shared__ int ri[32], si[8];
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

// 8 4-bit codes (element 2k in the low nibble of byte k) dotted with 8 floats.
__device__ __forceinline__ float dot8_u4(unsigned int p, float4 a, float4 b) {
    return a.x * (float)(p & 15u) + a.y * (float)((p >> 4) & 15u) + a.z * (float)((p >> 8) & 15u) +
           a.w * (float)((p >> 12) & 15u) + b.x * (float)((p >> 16) & 15u) + b.y * (float)((p >> 20) & 15u) +
           b.z * (float)((p >> 24) & 15u) + b.w * (float)(p >> 28);
}

__device__ __forceinline__ float sum8(float4 a, float4 b) { return a.x + a.y + a.z + a.w + b.x + b.y + b.z + b.w; }

// The picks of a layer, grouped by expert: unique expert u has device row eptr[u] (one 4-bit
// expert, layout of aegist/hybrid.py expert_layout) and picks [ustart[u], ustart[u+1]);
// pick p is token ptok[p], originally pick psrc[p] (= token * K + rank), weight wts[psrc[p]].
// Groups of 64 inputs; weight = code * scale + min, scale and min f16 per group.

// SwiGLU input side for few picks per expert: act[p, r] = silu(gate_r . x) * (up_r . x), where
// gate_up rows are [gate (I rows); up (I rows)] of D inputs. One warp per r; grid (I/8, U).
extern "C" __global__ void moe_gu_gemv(const float* __restrict__ h, const unsigned long long* __restrict__ eptr,
                                       const int* __restrict__ ustart, const int* __restrict__ ptok,
                                       float* __restrict__ act, int D, int I, int offs, int offm) {
    int lane = threadIdx.x & 31, r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), u = blockIdx.y;
    if (r >= I) return;
    const unsigned char* base = (const unsigned char*)eptr[u];
    const uint4* qg = (const uint4*)(base + (size_t)r * (D / 2));
    const uint4* qu = (const uint4*)(base + (size_t)(I + r) * (D / 2));
    const unsigned short* sc = (const unsigned short*)(base + offs);
    const unsigned short* mn = (const unsigned short*)(base + offm);
    int G = D / 64;
    for (int p = ustart[u]; p < ustart[u + 1]; p++) {
        const float* x = h + (size_t)ptok[p] * D;
        float ag = 0.f, au = 0.f;
        for (int g = lane; g < G; g += 32) {
            const float4* xv = (const float4*)(x + g * 64);
            float sx = 0.f, dg = 0.f, du = 0.f;
#pragma unroll
            for (int c = 0; c < 2; c++) {
                uint4 wg = __ldg(qg + g * 2 + c), wu = __ldg(qu + g * 2 + c);
                float4 x0 = xv[c * 8], x1 = xv[c * 8 + 1], x2 = xv[c * 8 + 2], x3 = xv[c * 8 + 3];
                float4 x4 = xv[c * 8 + 4], x5 = xv[c * 8 + 5], x6 = xv[c * 8 + 6], x7 = xv[c * 8 + 7];
                dg += dot8_u4(wg.x, x0, x1) + dot8_u4(wg.y, x2, x3) + dot8_u4(wg.z, x4, x5) + dot8_u4(wg.w, x6, x7);
                du += dot8_u4(wu.x, x0, x1) + dot8_u4(wu.y, x2, x3) + dot8_u4(wu.z, x4, x5) + dot8_u4(wu.w, x6, x7);
                sx += sum8(x0, x1) + sum8(x2, x3) + sum8(x4, x5) + sum8(x6, x7);
            }
            size_t gi = (size_t)r * G + g, gu = (size_t)(I + r) * G + g;
            ag += h2f(sc[gi]) * dg + h2f(mn[gi]) * sx;
            au += h2f(sc[gu]) * du + h2f(mn[gu]) * sx;
        }
        ag = warp_sum(ag);
        au = warp_sum(au);
        if (lane == 0) act[(size_t)p * I + r] = silu(ag) * au;
    }
}

// SwiGLU output side for few picks per expert: ypick[psrc[p], o] = wts[psrc[p]] * (down_o . act[p]).
// down rows have I inputs; requires I == 512 (16 uint4 per row). Each warp does two rows (one per
// half-warp); grid (D/16, U), 256 threads.
extern "C" __global__ void moe_dn_gemv(const float* __restrict__ act, const unsigned long long* __restrict__ eptr,
                                       const int* __restrict__ ustart, const int* __restrict__ psrc,
                                       const float* __restrict__ wts, float* __restrict__ ypick, int D, int I, int offq,
                                       int offs, int offm) {
    int lane = threadIdx.x & 31, half = lane >> 4, sub = lane & 15, u = blockIdx.y;
    int o = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 2 + half;
    bool ok = o < D;
    int oo = ok ? o : 0;
    const unsigned char* base = (const unsigned char*)eptr[u];
    uint4 w = __ldg((const uint4*)(base + offq + (size_t)oo * (I / 2)) + sub);
    size_t gi = (size_t)oo * (I / 64) + (sub >> 1);
    float s = h2f(((const unsigned short*)(base + offs))[gi]), m = h2f(((const unsigned short*)(base + offm))[gi]);
    for (int p = ustart[u]; p < ustart[u + 1]; p++) {
        const float4* a = (const float4*)(act + (size_t)p * I + sub * 32);
        float4 a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
        float d = dot8_u4(w.x, a0, a1) + dot8_u4(w.y, a2, a3) + dot8_u4(w.z, a4, a5) + dot8_u4(w.w, a6, a7);
        float v = s * d + m * (sum8(a0, a1) + sum8(a2, a3) + sum8(a4, a5) + sum8(a6, a7));
        v += __shfl_xor_sync(0xffffffffu, v, 8);
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        if (ok && sub == 0) {
            int src = psrc[p];
            ypick[(size_t)src * D + o] = v * wts[src];
        }
    }
}

// Dequantise rows [row0, row0 + n) x one 64-value group (index grp) of a 4-bit matrix into
// Ws[n][65]. Row stride rs bytes; the scale and min tables have G groups per row. n <= 64.
__device__ __forceinline__ void load_q4_tile(float (*Ws)[65], const unsigned char* q, const unsigned short* sc,
                                             const unsigned short* mn, int row0, int n, int rs, int G, int grp) {
    for (int i = threadIdx.x; i < n * 8; i += blockDim.x) {
        int r = i >> 3, c = i & 7;  // 8 uint32 (64 codes) per row
        int row = row0 + r;
        unsigned int p = *(const unsigned int*)(q + (size_t)row * rs + grp * 32 + c * 4);
        float s = h2f(sc[(size_t)row * G + grp]), m = h2f(mn[(size_t)row * G + grp]);
#pragma unroll
        for (int b = 0; b < 8; b++) Ws[r][c * 8 + b] = (float)((p >> (4 * b)) & 15u) * s + m;
    }
}

// SwiGLU input side for many picks per expert (prefill). Grid (I/32, U), 256 threads. A block
// computes 32 units (gate rows r0.., up rows I+r0..) for all of expert u's picks, 32 at a time.
extern "C" __global__ void moe_gu_gemm(const float* __restrict__ h, const unsigned long long* __restrict__ eptr,
                                       const int* __restrict__ ustart, const int* __restrict__ ptok,
                                       float* __restrict__ act, int D, int I, int offs, int offm) {
    __shared__ float Xs[32][65];
    __shared__ float Ws[64][65];
    int u = blockIdx.y, r0 = blockIdx.x * 32, tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const unsigned char* base = (const unsigned char*)eptr[u];
    const unsigned short* sc = (const unsigned short*)(base + offs);
    const unsigned short* mn = (const unsigned short*)(base + offm);
    int G = D / 64, p_end = ustart[u + 1];
    for (int p0 = ustart[u]; p0 < p_end; p0 += 32) {
        float ag[2][2] = {{0.f, 0.f}, {0.f, 0.f}}, au[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
        for (int grp = 0; grp < G; grp++) {
            for (int i = tid; i < 32 * 64; i += blockDim.x) {
                int pr = i >> 6, c = i & 63, p = p0 + pr;
                Xs[pr][c] = p < p_end ? h[(size_t)ptok[p] * D + grp * 64 + c] : 0.f;
            }
            load_q4_tile(Ws, base, sc, mn, r0, 32, D / 2, G, grp);
            load_q4_tile(Ws + 32, base, sc, mn, I + r0, 32, D / 2, G, grp);
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
extern "C" __global__ void moe_dn_gemm(const float* __restrict__ act, const unsigned long long* __restrict__ eptr,
                                       const int* __restrict__ ustart, const int* __restrict__ psrc,
                                       const float* __restrict__ wts, float* __restrict__ ypick, int D, int I, int offq,
                                       int offs, int offm) {
    __shared__ float Xs[32][65];
    __shared__ float Ws[64][65];
    int u = blockIdx.y, o0 = blockIdx.x * 64, tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const unsigned char* base = (const unsigned char*)eptr[u];
    const unsigned short* sc = (const unsigned short*)(base + offs);
    const unsigned short* mn = (const unsigned short*)(base + offm);
    int G = I / 64, p_end = ustart[u + 1];
    for (int p0 = ustart[u]; p0 < p_end; p0 += 32) {
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int grp = 0; grp < G; grp++) {
            for (int i = tid; i < 32 * 64; i += blockDim.x) {
                int pr = i >> 6, c = i & 63, p = p0 + pr;
                Xs[pr][c] = p < p_end ? act[(size_t)p * I + grp * 64 + c] : 0.f;
            }
            load_q4_tile(Ws, base + offq, sc, mn, o0, 64, I / 2, G, grp);
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

// x[t, d] += the sum over the token's K picks of ypick[t*K + j, d]. Grid (T, D/256), block 256.
extern "C" __global__ void moe_reduce(const float* __restrict__ ypick, float* __restrict__ x, int D, int K) {
    int t = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float s = 0.f;
    for (int j = 0; j < K; j++) s += ypick[((size_t)t * K + j) * D + d];
    x[(size_t)t * D + d] += s;
}

// Shared expert: x[t] += sigmoid(h[t] . wg) * y[t]. One block per token.
extern "C" __global__ void shared_gate_add(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ h,
                                           const bf16_t* __restrict__ wg, int D) {
    __shared__ float red[32];
    int t = blockIdx.x;
    const float* hr = h + (size_t)t * D;
    float a = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) a += hr[i] * bf2f(wg[i]);
    float g = sigmoidf_(block_sum(a, red));
    for (int i = threadIdx.x; i < D; i += blockDim.x) x[(size_t)t * D + i] += g * y[(size_t)t * D + i];
}
