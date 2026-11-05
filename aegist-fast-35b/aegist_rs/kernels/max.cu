// 3-bit experts for the 2.4T brain (aegist/fetch_max.py, format "q3g64").
//
// One expert record: [gate codes][gate scales][up codes][up scales][down codes][down scales].
// A matrix [rows, K] has K/64 groups per row; a group is 24 bytes of codes (64 codes of 3 bits,
// a little-endian bit stream: code i at bit 3*i) and one f16 scale; weight = scale * (code - 4).
// So gate.x = sum over groups of scale * (codes . x - 4 * sum(x)).

// The 8 codes of 24-bit word w dotted with 8 floats.
__device__ __forceinline__ float dot8_u3(unsigned int w, float4 a, float4 b) {
    return a.x * (float)(w & 7u) + a.y * (float)((w >> 3) & 7u) + a.z * (float)((w >> 6) & 7u) +
           a.w * (float)((w >> 9) & 7u) + b.x * (float)((w >> 12) & 7u) + b.y * (float)((w >> 15) & 7u) +
           b.z * (float)((w >> 18) & 7u) + b.w * (float)((w >> 21) & 7u);
}

// The 8 24-bit words of one group (24 bytes, 8-byte aligned) from its 6 32-bit words.
__device__ __forceinline__ void words_q3(const unsigned char* g, unsigned int (&c)[8]) {
    uint2 p0 = __ldg((const uint2*)g), p1 = __ldg((const uint2*)(g + 8)), p2 = __ldg((const uint2*)(g + 16));
    unsigned int w0 = p0.x, w1 = p0.y, w2 = p1.x, w3 = p1.y, w4 = p2.x, w5 = p2.y;
    c[0] = w0 & 0xFFFFFFu;
    c[1] = ((w0 >> 24) | (w1 << 8)) & 0xFFFFFFu;
    c[2] = ((w1 >> 16) | (w2 << 16)) & 0xFFFFFFu;
    c[3] = w2 >> 8;
    c[4] = w3 & 0xFFFFFFu;
    c[5] = ((w3 >> 24) | (w4 << 8)) & 0xFFFFFFu;
    c[6] = ((w4 >> 16) | (w5 << 16)) & 0xFFFFFFu;
    c[7] = w5 >> 8;
}

// scale * (codes . x - 4 * sum(x)) for one group; x is 64 floats (16-byte aligned).
__device__ __forceinline__ float group_dot_q3(const unsigned char* g, unsigned short sc, const float* x) {
    unsigned int c[8];
    words_q3(g, c);
    const float4* xv = (const float4*)x;
    float d = 0.f, sx = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        float4 a = xv[2 * j], b = xv[2 * j + 1];
        d += dot8_u3(c[j], a, b);
        sx += a.x + a.y + a.z + a.w + b.x + b.y + b.z + b.w;
    }
    return h2f(sc) * (d - 4.f * sx);
}

// SwiGLU input side for few picks per expert: act[p, r] = silu(gate_r . x) * (up_r . x).
// Offsets (bytes, inside a record): gate codes 0, gate scales gs, up codes uq, up scales us.
// One warp per r; grid (I/8, U), 256 threads.
extern "C" __global__ void moe3_gu_gemv(const float* __restrict__ h, const unsigned long long* __restrict__ eptr,
                                        const int* __restrict__ ustart, const int* __restrict__ ptok,
                                        float* __restrict__ act, int D, int I, int gs, int uq, int us) {
    int lane = threadIdx.x & 31, r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), u = blockIdx.y;
    if (r >= I) return;
    const unsigned char* base = (const unsigned char*)eptr[u];
    int G = D / 64;
    const unsigned char* qg = base + (size_t)r * G * 24;
    const unsigned char* qu = base + uq + (size_t)r * G * 24;
    const unsigned short* sg = (const unsigned short*)(base + gs) + (size_t)r * G;
    const unsigned short* su = (const unsigned short*)(base + us) + (size_t)r * G;
    for (int p = ustart[u]; p < ustart[u + 1]; p++) {
        const float* x = h + (size_t)ptok[p] * D;
        float ag = 0.f, au = 0.f;
        for (int g = lane; g < G; g += 32) {
            ag += group_dot_q3(qg + g * 24, sg[g], x + g * 64);
            au += group_dot_q3(qu + g * 24, su[g], x + g * 64);
        }
        ag = warp_sum(ag);
        au = warp_sum(au);
        if (lane == 0) act[(size_t)p * I + r] = silu(ag) * au;
    }
}

// SwiGLU output side for few picks per expert: ypick[psrc[p], o] = wts[psrc[p]] * (down_o . act[p]).
// Down codes at dq, scales at ds. One warp per output row; grid (D/8, U), 256 threads.
extern "C" __global__ void moe3_dn_gemv(const float* __restrict__ act, const unsigned long long* __restrict__ eptr,
                                        const int* __restrict__ ustart, const int* __restrict__ psrc,
                                        const float* __restrict__ wts, float* __restrict__ ypick, int D, int I, int dq,
                                        int ds) {
    int lane = threadIdx.x & 31, o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), u = blockIdx.y;
    if (o >= D) return;
    const unsigned char* base = (const unsigned char*)eptr[u];
    int G = I / 64;
    const unsigned char* q = base + dq + (size_t)o * G * 24;
    const unsigned short* s = (const unsigned short*)(base + ds) + (size_t)o * G;
    for (int p = ustart[u]; p < ustart[u + 1]; p++) {
        const float* a = act + (size_t)p * I;
        float v = 0.f;
        for (int g = lane; g < G; g += 32) v += group_dot_q3(q + g * 24, s[g], a + g * 64);
        v = warp_sum(v);
        if (lane == 0) {
            int src = psrc[p];
            ypick[(size_t)src * D + o] = v * wts[src];
        }
    }
}

// Dequantise rows [row0, row0 + n) x group grp of a 3-bit matrix (G groups per row) into Ws[n][65].
__device__ __forceinline__ void load_q3_tile(float (*Ws)[65], const unsigned char* q, const unsigned short* sc,
                                             int row0, int n, int G, int grp) {
    for (int i = threadIdx.x; i < n * 8; i += blockDim.x) {
        int r = i >> 3, j = i & 7;  // 8 words of 8 codes per row
        int row = row0 + r;
        const unsigned char* g = q + ((size_t)row * G + grp) * 24 + j * 3;
        unsigned int w = (unsigned int)g[0] | ((unsigned int)g[1] << 8) | ((unsigned int)g[2] << 16);
        float s = h2f(sc[(size_t)row * G + grp]);
#pragma unroll
        for (int b = 0; b < 8; b++) Ws[r][j * 8 + b] = ((float)((w >> (3 * b)) & 7u) - 4.f) * s;
    }
}

// SwiGLU input side for many picks per expert (prefill). Grid (I/32, U), 256 threads.
extern "C" __global__ void moe3_gu_gemm(const float* __restrict__ h, const unsigned long long* __restrict__ eptr,
                                        const int* __restrict__ ustart, const int* __restrict__ ptok,
                                        float* __restrict__ act, int D, int I, int gs, int uq, int us) {
    __shared__ float Xs[32][65];
    __shared__ float Ws[64][65];
    int u = blockIdx.y, r0 = blockIdx.x * 32, tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const unsigned char* base = (const unsigned char*)eptr[u];
    const unsigned short* sg = (const unsigned short*)(base + gs);
    const unsigned short* su = (const unsigned short*)(base + us);
    int G = D / 64, p_end = ustart[u + 1];
    for (int p0 = ustart[u]; p0 < p_end; p0 += 32) {
        float ag[2][2] = {{0.f, 0.f}, {0.f, 0.f}}, au[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
        for (int grp = 0; grp < G; grp++) {
            for (int i = tid; i < 32 * 64; i += blockDim.x) {
                int pr = i >> 6, c = i & 63, p = p0 + pr;
                Xs[pr][c] = p < p_end ? h[(size_t)ptok[p] * D + grp * 64 + c] : 0.f;
            }
            load_q3_tile(Ws, base, sg, r0, 32, G, grp);
            load_q3_tile(Ws + 32, base + uq, su, r0, 32, G, grp);
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
extern "C" __global__ void moe3_dn_gemm(const float* __restrict__ act, const unsigned long long* __restrict__ eptr,
                                        const int* __restrict__ ustart, const int* __restrict__ psrc,
                                        const float* __restrict__ wts, float* __restrict__ ypick, int D, int I, int dq,
                                        int ds) {
    __shared__ float Xs[32][65];
    __shared__ float Ws[64][65];
    int u = blockIdx.y, o0 = blockIdx.x * 64, tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const unsigned char* base = (const unsigned char*)eptr[u];
    const unsigned short* sc = (const unsigned short*)(base + ds);
    int G = I / 64, p_end = ustart[u + 1];
    for (int p0 = ustart[u]; p0 < p_end; p0 += 32) {
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int grp = 0; grp < G; grp++) {
            for (int i = tid; i < 32 * 64; i += blockDim.x) {
                int pr = i >> 6, c = i & 63, p = p0 + pr;
                Xs[pr][c] = p < p_end ? act[(size_t)p * I + grp * 64 + c] : 0.f;
            }
            load_q3_tile(Ws, base + dq, sc, o0, 64, G, grp);
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
