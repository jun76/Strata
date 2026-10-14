// Qwen3.6 decode math ported from src/qwen35/q36ref.cu (plain reference path).
#include <metal_stdlib>
using namespace metal;
enum : int { T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_BF16 = 30 };

inline float h2f(ushort h) {
    const uint s = (uint)(h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) {
        const float v = ldexp((float)m, -24);
        return s ? -v : v;
    }
    const uint f = e == 31 ? (s | 0x7f800000u | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
    return as_type<float>(f);
}
inline ushort ld16(const device uchar *p) { return (ushort)(p[0] | (p[1] << 8)); }

inline void scale_min_k4(int j, const device uchar *q, thread int &d, thread int &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/// The 32 values of chunk `c` of one row (`row` = the row's first byte).
template <int T> inline void deq32(const device uchar *row, int c, thread float *out) {
    if constexpr (T == T_F32) {
        const device float *r = (const device float *)row + 32 * c;
        for (int l = 0; l < 32; ++l)
            out[l] = r[l];
    } else if constexpr (T == T_F16) {
        const device uchar *r = row + 64 * c;
        for (int l = 0; l < 32; ++l)
            out[l] = h2f(ld16(r + 2 * l));
    } else if constexpr (T == T_BF16) {
        const device uchar *r = row + 64 * c;
        for (int l = 0; l < 32; ++l)
            out[l] = as_type<float>((uint)ld16(r + 2 * l) << 16);
    } else if constexpr (T == T_Q8_0) {
        const device uchar *b = row + 34 * c;
        const float d = h2f(ld16(b));
        const device char *q = (const device char *)(b + 2);
        for (int l = 0; l < 32; ++l)
            out[l] = d * (float)q[l];
    } else if constexpr (T == T_Q4_K) {
        const device uchar *b = row + 144 * (c >> 3);
        const int j = c & 7;
        const float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
        int sc, m;
        scale_min_k4(j, b + 4, sc, m);
        const float d1 = d * (float)sc, m1 = dmin * (float)m;
        const device uchar *qs = b + 16 + 32 * (j >> 1);
        for (int l = 0; l < 32; ++l)
            out[l] = d1 * (float)((j & 1) ? (qs[l] >> 4) : (qs[l] & 0xF)) - m1;
    } else if constexpr (T == T_Q5_K) {
        const device uchar *b = row + 176 * (c >> 3);
        const int j = c & 7;
        const float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
        int sc, m;
        scale_min_k4(j, b + 4, sc, m);
        const float d1 = d * (float)sc, m1 = dmin * (float)m;
        const device uchar *qh = b + 16;
        const device uchar *qs = b + 48 + 32 * (j >> 1);
        for (int l = 0; l < 32; ++l) {
            const int q = ((j & 1) ? (qs[l] >> 4) : (qs[l] & 0xF)) + ((qh[l] >> j) & 1 ? 16 : 0);
            out[l] = d1 * (float)q - m1;
        }
    } else if constexpr (T == T_Q6_K) {
        const device uchar *b = row + 210 * (c >> 3);
        const int j = c & 7, n = j >> 2, r = j & 3;
        const device uchar *ql = b + 64 * n;
        const device uchar *qh = b + 128 + 32 * n;
        const device char *sc = (const device char *)(b + 192) + 8 * n;
        const float d = h2f(ld16(b + 208));
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int q;
            if (r == 0)
                q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (r == 1)
                q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (r == 2)
                q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else
                q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            out[l] = d * (float)sc[is + 2 * r] * (float)(q - 32);
        }
    }
}

inline float block_sum(float v, threadgroup float *sh, uint tid, uint nt) {
    sh[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nt / 2; s > 0; s >>= 1) {
        if (tid < s)
            sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

#define kThreads 128

/// y[sel][o] = W_e[o] . x[sel] for row o (grid.x), selection `sel` (grid.y): expert ids[sel] (or 0 without ids),
/// x of selection sel at x + sel * x_stride.  `beta` != 0 adds into y.
template <int T>
kernel void q36_gemv(const device uchar *W [[buffer(0)]], constant ulong &row_bytes [[buffer(1)]],
                     constant ulong &expert_bytes [[buffer(2)]], constant int &unused_ids [[buffer(3)]],
                     constant int &n_in [[buffer(4)]], const device float *x [[buffer(5)]],
                     constant ulong &x_stride [[buffer(6)]], device float *y [[buffer(7)]],
                     constant int &n_out [[buffer(8)]], uint tid [[thread_index_in_threadgroup]],
                     uint3 bid [[threadgroup_position_in_grid]], uint3 ng [[threadgroups_per_grid]],
                     uint3 nt3 [[threads_per_threadgroup]]) {
    const uint nt = nt3.x;
    threadgroup float sh[kThreads];
    const int o = bid.x, sel = bid.y;
    const int e = expert_bytes ? sel : 0;
    const device uchar *row = W + (ulong)e * expert_bytes + (ulong)o * row_bytes;
    const device float *xs = x + (ulong)sel * x_stride;
    float acc = 0.f, v[32];
    for (int c = tid; c < n_in / 32; c += nt) {
        deq32<T>(row, c, v);
        const device float *xc = xs + 32 * c;
        for (int l = 0; l < 32; ++l)
            acc += v[l] * xc[l];
    }
    acc = block_sum(acc, sh, tid, nt);
    if (tid == 0)
        y[(ulong)sel * n_out + o] = acc;
}

template <int T>
kernel void q36_row(const device uchar *W [[buffer(0)]], constant ulong &row_bytes [[buffer(1)]],
                    constant int &row [[buffer(2)]], constant int &n [[buffer(3)]], device float *out [[buffer(4)]],
                    uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
                    uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]) {
    const uint nt = nt3.x;
    const device uchar *r = W + (ulong)row * row_bytes;
    for (int c = bid.x * nt + tid; c < n / 32; c += ng.x * nt) {
        float vals[32];
        deq32<T>(r, c, vals);
        for (int j = 0; j < 32; ++j)
            out[32 * c + j] = vals[j];
    }
}

/// y = x / sqrt(mean(x^2) + eps) * w, per group of n (grid = groups); w (length n) may be null.
kernel void q36_rms(const device float *x [[buffer(0)]], const device float *w [[buffer(1)]],
                    device float *y [[buffer(2)]], constant int &n [[buffer(3)]], constant float &eps [[buffer(4)]],
                    uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
                    uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]) {
    const uint nt = nt3.x;
    threadgroup float sh[kThreads];
    const device float *xg = x + (ulong)bid.x * n;
    device float *yg = y + (ulong)bid.x * n;
    float s = 0.f;
    for (int i = tid; i < n; i += nt)
        s += xg[i] * xg[i];
    s = block_sum(s, sh, tid, nt);
    const float r = 1.0f / sqrt(s / (float)n + eps);
    for (int i = tid; i < n; i += nt)
        yg[i] = xg[i] * r * (w ? w[i] : 1.0f);
}

inline float silu(float v) { return v / (1.0f + exp(-v)); }
inline float sigm(float v) { return 1.0f / (1.0f + exp(-v)); }
inline float softplus(float v) { return v > 20.f ? v : log(1.0f + exp(v)); }

/// The causal 4-tap conv over the 8192 qkv channels with its 3-input state, then SiLU; the state moves by one.
kernel void q36_conv(const device float *qkv [[buffer(0)]], device float *state [[buffer(1)]],
                     const device float *w [[buffer(2)]], device float *out [[buffer(3)]],
                     constant int &C [[buffer(4)]], constant int &K [[buffer(5)]],
                     uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
                     uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]) {
    const uint nt = nt3.x;
    const int c = bid.x * nt + tid;
    if (c >= C)
        return;
    float acc = 0.f;
    for (int k = 0; k < K - 1; ++k)
        acc += state[(ulong)k * C + c] * w[(ulong)c * K + k];
    acc += qkv[c] * w[(ulong)c * K + K - 1];
    for (int k = 0; k < K - 2; ++k)
        state[(ulong)k * C + c] = state[(ulong)(k + 1) * C + c];
    state[(ulong)(K - 2) * C + c] = qkv[c];
    out[c] = silu(acc);
}

/// One gated delta-rule step per V head (grid = HV, block = D = 128): S[i][j] row i owned by thread i.
kernel void q36_gdn(const device float *conv [[buffer(0)]], const device float *alpha [[buffer(1)]],
                    const device float *dt_bias [[buffer(2)]], const device float *ssm_a [[buffer(3)]],
                    const device float *beta_raw [[buffer(4)]], device float *S [[buffer(5)]],
                    device float *out [[buffer(6)]], constant int &HK [[buffer(7)]], constant int &HV [[buffer(8)]],
                    constant int &D [[buffer(9)]], constant float &eps [[buffer(10)]],
                    uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
                    uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]) {
    const uint nt = nt3.x;
    threadgroup float sm[384];
    threadgroup float *q = sm;
    threadgroup float *k = sm + D;
    threadgroup float *red = sm + 2 * D;
    const int h = bid.x, kh = h % HK, i = tid;
    const float qi = conv[kh * D + i], ki = conv[HK * D + kh * D + i], vi = conv[2 * HK * D + h * D + i];
    const float qn = block_sum(qi * qi, red, tid, nt), kn = block_sum(ki * ki, red, tid, nt);
    q[i] = qi / sqrt(qn + eps) / sqrt((float)D);
    k[i] = ki / sqrt(kn + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float g = exp(softplus(alpha[h] + dt_bias[h]) * ssm_a[h]);
    const float b = sigm(beta_raw[h]);
    device float *Si = S + ((ulong)h * D + i) * D;
    float sk = 0.f;
    for (int j = 0; j < D; ++j) {
        Si[j] *= g;
        sk += Si[j] * k[j];
    }
    const float d = (vi - sk) * b;
    float o = 0.f;
    for (int j = 0; j < D; ++j) {
        Si[j] += d * k[j];
        o += Si[j] * q[j];
    }
    out[h * D + i] = o;
}

/// The fused path's gate and beta, in place: alpha -> softplus(alpha + dt) * ssm_a (exp'd by the step), beta ->
/// sigmoid.
kernel void q36_gdn_gb(device float *alpha [[buffer(0)]], const device float *dt_bias [[buffer(1)]],
                       const device float *ssm_a [[buffer(2)]], device float *beta [[buffer(3)]],
                       constant int &HV [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
                       uint3 bid [[threadgroup_position_in_grid]], uint3 ng [[threadgroups_per_grid]],
                       uint3 nt3 [[threads_per_threadgroup]]) {
    const uint nt = nt3.x;
    const int h = tid;
    if (h >= HV)
        return;
    alpha[h] = softplus(alpha[h] + dt_bias[h]) * ssm_a[h];
    beta[h] = sigm(beta[h]);
}

/// o = rms(o) * w * silu(z), per V head (grid = HV, block = D).
kernel void q36_gated_norm(device float *o [[buffer(0)]], const device float *w [[buffer(1)]],
                           const device float *z [[buffer(2)]], constant int &D [[buffer(3)]],
                           constant float &eps [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
                           uint3 bid [[threadgroup_position_in_grid]], uint3 ng [[threadgroups_per_grid]],
                           uint3 nt3 [[threads_per_threadgroup]]) {
    const uint nt = nt3.x;
    threadgroup float sh[kThreads];
    device float *oh = o + (ulong)bid.x * D;
    const float v = oh[tid];
    const float s = block_sum(v * v, sh, tid, nt);
    oh[tid] = v / sqrt(s / (float)D + eps) * w[tid] * silu(z[bid.x * D + tid]);
}

/// The attention layer's per-token preparation: q (from the [q | gate] pairs) and k RMS-normed per head with their
/// weights, NEOX RoPE over the first n_rot dims, k and v appended to the cache at `pos`, the gate kept.
kernel void q36_attn_prep(const device float *qg [[buffer(0)]], const device float *kx [[buffer(1)]],
                          const device float *vx [[buffer(2)]], const device float *qw [[buffer(3)]],
                          const device float *kw [[buffer(4)]], device float *q [[buffer(5)]],
                          device float *gate [[buffer(6)]], device float *kc [[buffer(7)]],
                          device float *vc [[buffer(8)]], constant int &H [[buffer(9)]],
                          constant int &HKV [[buffer(10)]], constant int &D [[buffer(11)]],
                          constant int &n_rot [[buffer(12)]], constant int &pos [[buffer(13)]],
                          constant int &ctx [[buffer(14)]], constant float &base [[buffer(15)]],
                          constant float &eps [[buffer(16)]], uint tid [[thread_index_in_threadgroup]],
                          uint3 bid [[threadgroup_position_in_grid]], uint3 ng [[threadgroups_per_grid]],
                          uint3 nt3 [[threads_per_threadgroup]]) {
    const uint nt = nt3.x;
    threadgroup float sh[kThreads];
    threadgroup float buf[512];
    const int head = bid.x; // 0..H-1: query heads, H..H+HKV-1: key heads
    const bool isq = head < H;
    const device float *src = isq ? qg + (ulong)head * 2 * D : kx + (ulong)(head - H) * D;
    float s = 0.f;
    for (int i = tid; i < D; i += nt)
        s += src[i] * src[i];
    s = block_sum(s, sh, tid, nt);
    const float r = 1.0f / sqrt(s / (float)D + eps);
    for (int i = tid; i < D; i += nt)
        buf[i] = src[i] * r * (isq ? qw[i] : kw[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *dst = isq ? q + (ulong)head * D : kc + ((ulong)(head - H) * ctx + pos) * D;
    const int rot_half = n_rot / 2;
    for (int i = tid; i < D; i += nt) {
        float v = buf[i];
        if (i < n_rot) {
            const int p = i < rot_half ? i : i - rot_half;
            const float theta = (float)pos * pow(base, -2.0f * (float)p / (float)n_rot);
            const float cs = cos(theta), sn = sin(theta);
            v = i < rot_half ? buf[i] * cs - buf[i + rot_half] * sn : buf[i - rot_half] * sn + buf[i] * cs;
        }
        dst[i] = v;
    }
    if (isq)
        for (int i = tid; i < D; i += nt)
            gate[(ulong)head * D + i] = qg[(ulong)head * 2 * D + D + i];
    else
        for (int i = tid; i < D; i += nt)
            vc[((ulong)(head - H) * ctx + pos) * D + i] = vx[(ulong)(head - H) * D + i];
}

/// softmax(q k^T * scale) v for one query head per block (GQA: KV head = head / (H / HKV)), times sigmoid(gate).
kernel void q36_attn(const device float *q [[buffer(0)]], const device float *kc [[buffer(1)]],
                     const device float *vc [[buffer(2)]], const device float *gate [[buffer(3)]],
                     device float *out [[buffer(4)]], constant int &H [[buffer(5)]], constant int &HKV [[buffer(6)]],
                     constant int &D [[buffer(7)]], constant int &n_pos [[buffer(8)]], constant int &ctx [[buffer(9)]],
                     constant float &scale [[buffer(10)]], uint tid [[thread_index_in_threadgroup]],
                     uint3 bid [[threadgroup_position_in_grid]], uint3 ng [[threadgroups_per_grid]],
                     uint3 nt3 [[threads_per_threadgroup]]) {
    const uint nt = nt3.x;
    threadgroup float sc[4096];
    threadgroup float sh[kThreads];
    const int h = bid.x, kvh = h / (H / HKV);
    const device float *qh = q + (ulong)h * D;
    for (int p = tid; p < n_pos; p += nt) {
        const device float *kp = kc + ((ulong)kvh * ctx + p) * D;
        float s = 0.f;
        for (int i = 0; i < D; ++i)
            s += qh[i] * kp[i];
        sc[p] = s * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -INFINITY;
    for (int p = tid; p < n_pos; p += nt)
        mx = max(mx, sc[p]);
    sh[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = nt / 2; s > 0; s >>= 1) {
        if ((int)tid < s)
            sh[tid] = max(sh[tid], sh[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mx = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.f;
    for (int p = tid; p < n_pos; p += nt) {
        sc[p] = exp(sc[p] - mx);
        sum += sc[p];
    }
    sum = block_sum(sum, sh, tid, nt);
    for (int i = tid; i < D; i += nt) {
        float acc = 0.f;
        for (int p = 0; p < n_pos; ++p)
            acc += sc[p] * vc[((ulong)kvh * ctx + p) * D + i];
        out[(ulong)h * D + i] = acc / sum * sigm(gate[(ulong)h * D + i]);
    }
}

kernel void q36_swiglu(const device float *g [[buffer(0)]], const device float *u [[buffer(1)]],
                       device float *h [[buffer(2)]], constant int &n [[buffer(3)]],
                       uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
                       uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]) {
    const uint nt = nt3.x;
    const int i = bid.x * nt + tid;
    if (i < n)
        h[i] = silu(g[i]) * u[i];
}
kernel void q36_add(device float *x [[buffer(0)]], const device float *y [[buffer(1)]], constant int &n [[buffer(2)]],
                    uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
                    uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]) {
    const uint nt = nt3.x;
    const int i = bid.x * nt + tid;
    if (i < n)
        x[i] += y[i];
}
/// x += sum_s w[s] * y[s] (the routed experts), + sigmoid(gs) * ys (the shared expert)
kernel void q36_moe_sum(device float *x [[buffer(0)]], const device float *y [[buffer(1)]],
                        const device float *w [[buffer(2)]], constant int &k [[buffer(3)]],
                        const device float *ys [[buffer(4)]], const device float *gs [[buffer(5)]],
                        constant int &n [[buffer(6)]], uint tid [[thread_index_in_threadgroup]],
                        uint3 bid [[threadgroup_position_in_grid]], uint3 ng [[threadgroups_per_grid]],
                        uint3 nt3 [[threads_per_threadgroup]]) {
    const uint nt = nt3.x;
    const int i = bid.x * nt + tid;
    if (i >= n)
        return;
    float acc = 0.f;
    for (int s = 0; s < k; ++s)
        acc += w[s] * y[(ulong)s * n + i];
    x[i] += acc + sigm(gs[0]) * ys[i];
}

template [[host_name("q36_gemv_0")]] kernel void
q36_gemv<0>(const device uchar *W [[buffer(0)]], constant ulong &row_bytes [[buffer(1)]],
            constant ulong &expert_bytes [[buffer(2)]], constant int &unused_ids [[buffer(3)]],
            constant int &n_in [[buffer(4)]], const device float *x [[buffer(5)]],
            constant ulong &x_stride [[buffer(6)]], device float *y [[buffer(7)]], constant int &n_out [[buffer(8)]],
            uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
            uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]);
template [[host_name("q36_gemv_1")]] kernel void
q36_gemv<1>(const device uchar *W [[buffer(0)]], constant ulong &row_bytes [[buffer(1)]],
            constant ulong &expert_bytes [[buffer(2)]], constant int &unused_ids [[buffer(3)]],
            constant int &n_in [[buffer(4)]], const device float *x [[buffer(5)]],
            constant ulong &x_stride [[buffer(6)]], device float *y [[buffer(7)]], constant int &n_out [[buffer(8)]],
            uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
            uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]);
template [[host_name("q36_gemv_8")]] kernel void
q36_gemv<8>(const device uchar *W [[buffer(0)]], constant ulong &row_bytes [[buffer(1)]],
            constant ulong &expert_bytes [[buffer(2)]], constant int &unused_ids [[buffer(3)]],
            constant int &n_in [[buffer(4)]], const device float *x [[buffer(5)]],
            constant ulong &x_stride [[buffer(6)]], device float *y [[buffer(7)]], constant int &n_out [[buffer(8)]],
            uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
            uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]);
template [[host_name("q36_gemv_12")]] kernel void
q36_gemv<12>(const device uchar *W [[buffer(0)]], constant ulong &row_bytes [[buffer(1)]],
             constant ulong &expert_bytes [[buffer(2)]], constant int &unused_ids [[buffer(3)]],
             constant int &n_in [[buffer(4)]], const device float *x [[buffer(5)]],
             constant ulong &x_stride [[buffer(6)]], device float *y [[buffer(7)]], constant int &n_out [[buffer(8)]],
             uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
             uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]);
template [[host_name("q36_gemv_13")]] kernel void
q36_gemv<13>(const device uchar *W [[buffer(0)]], constant ulong &row_bytes [[buffer(1)]],
             constant ulong &expert_bytes [[buffer(2)]], constant int &unused_ids [[buffer(3)]],
             constant int &n_in [[buffer(4)]], const device float *x [[buffer(5)]],
             constant ulong &x_stride [[buffer(6)]], device float *y [[buffer(7)]], constant int &n_out [[buffer(8)]],
             uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
             uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]);
template [[host_name("q36_gemv_14")]] kernel void
q36_gemv<14>(const device uchar *W [[buffer(0)]], constant ulong &row_bytes [[buffer(1)]],
             constant ulong &expert_bytes [[buffer(2)]], constant int &unused_ids [[buffer(3)]],
             constant int &n_in [[buffer(4)]], const device float *x [[buffer(5)]],
             constant ulong &x_stride [[buffer(6)]], device float *y [[buffer(7)]], constant int &n_out [[buffer(8)]],
             uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
             uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]);
template [[host_name("q36_gemv_30")]] kernel void
q36_gemv<30>(const device uchar *W [[buffer(0)]], constant ulong &row_bytes [[buffer(1)]],
             constant ulong &expert_bytes [[buffer(2)]], constant int &unused_ids [[buffer(3)]],
             constant int &n_in [[buffer(4)]], const device float *x [[buffer(5)]],
             constant ulong &x_stride [[buffer(6)]], device float *y [[buffer(7)]], constant int &n_out [[buffer(8)]],
             uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
             uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]);
template [[host_name("q36_row_0")]] kernel void
q36_row<0>(const device uchar *W [[buffer(0)]], constant ulong &row_bytes [[buffer(1)]],
           constant int &row [[buffer(2)]], constant int &n [[buffer(3)]], device float *out [[buffer(4)]],
           uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
           uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]);
template [[host_name("q36_row_1")]] kernel void
q36_row<1>(const device uchar *W [[buffer(0)]], constant ulong &row_bytes [[buffer(1)]],
           constant int &row [[buffer(2)]], constant int &n [[buffer(3)]], device float *out [[buffer(4)]],
           uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
           uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]);
template [[host_name("q36_row_8")]] kernel void
q36_row<8>(const device uchar *W [[buffer(0)]], constant ulong &row_bytes [[buffer(1)]],
           constant int &row [[buffer(2)]], constant int &n [[buffer(3)]], device float *out [[buffer(4)]],
           uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
           uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]);
template [[host_name("q36_row_12")]] kernel void
q36_row<12>(const device uchar *W [[buffer(0)]], constant ulong &row_bytes [[buffer(1)]],
            constant int &row [[buffer(2)]], constant int &n [[buffer(3)]], device float *out [[buffer(4)]],
            uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
            uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]);
template [[host_name("q36_row_13")]] kernel void
q36_row<13>(const device uchar *W [[buffer(0)]], constant ulong &row_bytes [[buffer(1)]],
            constant int &row [[buffer(2)]], constant int &n [[buffer(3)]], device float *out [[buffer(4)]],
            uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
            uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]);
template [[host_name("q36_row_14")]] kernel void
q36_row<14>(const device uchar *W [[buffer(0)]], constant ulong &row_bytes [[buffer(1)]],
            constant int &row [[buffer(2)]], constant int &n [[buffer(3)]], device float *out [[buffer(4)]],
            uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
            uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]);
template [[host_name("q36_row_30")]] kernel void
q36_row<30>(const device uchar *W [[buffer(0)]], constant ulong &row_bytes [[buffer(1)]],
            constant int &row [[buffer(2)]], constant int &n [[buffer(3)]], device float *out [[buffer(4)]],
            uint tid [[thread_index_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
            uint3 ng [[threadgroups_per_grid]], uint3 nt3 [[threads_per_threadgroup]]);
