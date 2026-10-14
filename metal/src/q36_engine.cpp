// Experimental Qwen3.6-35B-A3B Metal decode engine. Reference math from src/qwen35/q36ref.cu.
// Dense weights stay in shared memory; only the selected experts are copied from the mapped GGUF.
#include "mtl_rt.h"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
namespace {
constexpr int kThreads = 128;
enum : int { T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_BF16 = 30 };
mtl::Context cx;
int lib;
std::map<std::string, mtl::Pipeline> pipes;
std::vector<mtl::Dispatch> pending;
std::vector<int> g_ids_host;
void flush() {
    if (!pending.empty()) {
        cx.run(pending);
        pending.clear();
    }
}
mtl::Arg arg(const mtl::Buffer &b) { return mtl::Arg::B(b); }
template <class T> mtl::Arg arg(const T &v) { return mtl::Arg::V(v); }
template <class... A> void dispatch(const std::string &name, int groups, int threads, const A &...args) {
    auto it = pipes.find(name);
    if (it == pipes.end())
        it = pipes.emplace(name, cx.pipeline(lib, name)).first;
    mtl::Dispatch d;
    d.p = &it->second;
    d.grid[0] = groups;
    d.tg[0] = threads;
    d.args = {arg(args)...};
    pending.push_back(std::move(d));
}
struct Tensor {
    mtl::Buffer d;
    const uint8_t *source = nullptr;
    int type = -1;
    std::vector<int64_t> shape;
    size_t row_bytes = 0;
};
size_t row_bytes_of(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry(type, be, bb))
        throw std::runtime_error("unsupported GGUF type");
    return (n / be) * bb;
}
void gemv(const Tensor &w, const mtl::Buffer &x, const mtl::Buffer &y, const mtl::Buffer *ids = nullptr, int n_sel = 1,
          size_t x_stride = 0) {
    const int n_in = w.shape[0], n_out = w.shape[1];
    const uint64_t eb = ids ? w.row_bytes * n_out : 0;
    if (ids) {
        for (int i = 0; i < n_sel; ++i)
            std::memcpy(static_cast<uint8_t *>(w.d.ptr) + i * eb, w.source + g_ids_host[i] * eb, eb);
    }
    auto it = pipes.find("q36_gemv_" + std::to_string(w.type));
    if (it == pipes.end())
        it = pipes.emplace("q36_gemv_" + std::to_string(w.type), cx.pipeline(lib, "q36_gemv_" + std::to_string(w.type)))
                 .first;
    mtl::Dispatch d;
    d.p = &it->second;
    d.grid[0] = n_out;
    d.grid[1] = n_sel;
    d.tg[0] = 128;
    d.args = {arg(w.d), arg((uint64_t)w.row_bytes), arg(eb), arg(0),    arg(n_in),
              arg(x),   arg((uint64_t)x_stride),    arg(y),  arg(n_out)};
    pending.push_back(std::move(d));
}
struct Model {
    strata::core::ModelGeometry g;
    int64_t top_k = 8;
    float eps = 1e-6f, rope_base = 1e7f;
    int n_rot = 64;
    std::map<std::string, Tensor> t;
    std::map<std::string, mtl::Buffer> f32;
    std::unique_ptr<strata::GgufModel> file; // the small vectors, f32

    const Tensor &W(const std::string &n) const {
        auto it = t.find(n);
        if (it == t.end()) {
            std::fprintf(stderr, "strata-q36-metal: missing tensor %s\n", n.c_str());
            std::exit(1);
        }
        return it->second;
    }
    const mtl::Buffer &F(const std::string &n) const {
        auto it = f32.find(n);
        if (it == f32.end()) {
            std::fprintf(stderr, "strata-q36-metal: missing vector %s\n", n.c_str());
            std::exit(1);
        }
        return it->second;
    }
};

float h2f_host(uint16_t h) {
    const uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0)
        return (s ? -1.f : 1.f) * std::ldexp((float)m, -24);
    const uint32_t f = e == 31 ? (s | 0x7f800000u | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
    float r;
    std::memcpy(&r, &f, 4);
    return r;
}

bool load(const std::string &path, Model &m, std::string &err) {
    const auto t0 = std::chrono::steady_clock::now();
    m.file = std::make_unique<strata::GgufModel>(strata::gguf_split_paths(path));
    const auto &gm = *m.file;
    if (!strata::core::geometry_from_gguf(gm.meta(), m.g, m.top_k, err))
        return false;
    if (m.g.family != strata::ModelFamily::Qwen35Moe) {
        err = "this reference path is for Qwen3.6-35B-A3B (qwen35moe) only";
        return false;
    }
    if (m.g.ssm_state_size != 128 || m.g.head_dim != 256 || m.g.ssm_d_conv != 4) {
        err = "Qwen3.6 Metal requires 128-wide GDN, 256-wide attention and a four-tap convolution";
        return false;
    }
    size_t checked = 0;
    if (!strata::core::check_gguf_layout(gm, m.g, err, &checked))
        return false;
    const std::string p = "qwen35moe.";
    if (const strata::MetaValue *v = gm.meta().get(p + "attention.layer_norm_rms_epsilon"))
        m.eps = (float)v->num();
    if (const strata::MetaValue *v = gm.meta().get(p + "rope.freq_base"))
        m.rope_base = (float)v->num();
    if (const strata::MetaValue *v = gm.meta().get(p + "rope.dimension_count"))
        m.n_rot = (int)v->u;
    if (m.n_rot <= 0 || m.n_rot > m.g.head_dim || m.n_rot % 2) {
        err = "invalid rotary dimension";
        return false;
    }
    uint64_t bytes = 0;
    for (size_t s = 0; s < gm.size(); ++s) {
        const strata::GgufFile &f = gm.shard(s);
        for (const strata::TensorInfo &ti : f.tensors()) {
            const uint64_t nb = strata::tensor_payload_bytes(ti);
            const uint8_t *src = f.tensor_data(ti);
            // the MTP block (blk.40) is not run by this path: not loaded
            if (ti.name.rfind("blk." + std::to_string(m.g.n_layers) + ".", 0) == 0)
                continue;
            const bool conv = ti.name.size() > 17 && ti.name.compare(ti.name.size() - 17, 17, "ssm_conv1d.weight") == 0;
            if (ti.shape.size() == 1 || conv) {
                // the small vectors (and the conv taps): f32 on the device
                std::vector<float> h(ti.elements());
                if (ti.type == T_F32)
                    std::memcpy(h.data(), src, h.size() * 4);
                else if (ti.type == T_BF16)
                    for (size_t i = 0; i < h.size(); ++i) {
                        const uint32_t u = (uint32_t)(src[2 * i] | (src[2 * i + 1] << 8)) << 16;
                        std::memcpy(&h[i], &u, 4);
                    }
                else if (ti.type == T_F16)
                    for (size_t i = 0; i < h.size(); ++i)
                        h[i] = h2f_host((uint16_t)(src[2 * i] | (src[2 * i + 1] << 8)));
                else {
                    err = ti.name + ": a 1-D tensor of type " + ti.type_name();
                    return false;
                }
                mtl::Buffer d = cx.alloc(h.size() * 4);
                std::memcpy(d.ptr, h.data(), h.size() * 4);
                m.f32[ti.name] = d;
                bytes += h.size() * 4;
                continue;
            }
            Tensor w;
            w.type = (int)ti.type;
            if (w.type != T_F32 && w.type != T_F16 && w.type != T_BF16 && w.type != T_Q8_0 && w.type != T_Q4_K &&
                w.type != T_Q5_K && w.type != T_Q6_K) {
                err = ti.name + ": unsupported Qwen3.6 Metal tensor type " + ti.type_name();
                return false;
            }
            for (uint64_t d : ti.shape)
                w.shape.push_back((int64_t)d);
            w.row_bytes = row_bytes_of(w.type, w.shape[0]);
            if (ti.shape.size() == 3) {
                w.source = src; // selected experts copied into a bounded per-tensor staging buffer
                w.d = cx.alloc(w.row_bytes * w.shape[1] * m.top_k);
            } else {
                w.d = cx.alloc(nb);
                std::memcpy(w.d.ptr, src, nb);
            }
            m.t[ti.name] = w;
            bytes += w.d.bytes;
        }
    }
    std::fprintf(stderr, "strata-q36-metal: %s: %zu tensors (layout checked), %.2f GiB on the GPU in %.1f s\n",
                 path.c_str(), checked, (double)bytes / (1 << 30),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

struct State {
    int ctx = 4096, pos = 0;
    // scratch
    mtl::Buffer x, h, y, qkv, z, alpha, beta, conv_out, gdn_out, qg, k, v, q, gate, att, router, eg, eu, eh, ey, sg, su,
        sh, sy, sgate, wts, logits;
    mtl::Buffer ids;
    std::vector<mtl::Buffer> conv_state, S, kc, vc; // per layer (GDN: conv + S; attention: K/V cache)
};

mtl::Buffer dalloc(size_t n) { return cx.alloc(n * 4); }

void init_state(const Model &m, State &s) {
    const auto &g = m.g;
    const int N = (int)g.n_embd, FF = (int)g.n_ff, K = (int)m.top_k;
    s.x = dalloc(N);
    s.h = dalloc(N);
    s.y = dalloc(N);
    s.qkv = dalloc(g.ssm_conv_channels);
    s.z = dalloc(g.ssm_value_dim);
    s.alpha = dalloc(g.ssm_v_heads);
    s.beta = dalloc(g.ssm_v_heads);
    s.conv_out = dalloc(g.ssm_conv_channels);
    s.gdn_out = dalloc(g.ssm_value_dim);
    s.qg = dalloc(2 * g.n_head * g.head_dim);
    s.k = dalloc(g.n_head_kv * g.head_dim);
    s.v = dalloc(g.n_head_kv * g.head_dim);
    s.q = dalloc(g.n_head * g.head_dim);
    s.gate = dalloc(g.n_head * g.head_dim);
    s.att = dalloc(g.n_head * g.head_dim);
    s.router = dalloc(g.n_expert);
    s.eg = dalloc((size_t)K * FF);
    s.eu = dalloc((size_t)K * FF);
    s.eh = dalloc((size_t)K * FF);
    s.ey = dalloc((size_t)K * N);
    s.sg = dalloc(g.n_ff_shexp);
    s.su = dalloc(g.n_ff_shexp);
    s.sh = dalloc(g.n_ff_shexp);
    s.sy = dalloc(N);
    s.sgate = dalloc(1);
    s.wts = dalloc(K);
    s.logits = dalloc(g.n_vocab);
    s.ids = cx.alloc(K * sizeof(int));
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (strata::core::is_qsa_layer(g, l)) {
            s.conv_state.push_back({});
            s.S.push_back({});
            s.kc.push_back(dalloc((size_t)g.n_head_kv * s.ctx * g.head_dim));
            s.vc.push_back(dalloc((size_t)g.n_head_kv * s.ctx * g.head_dim));
        } else {
            s.conv_state.push_back(dalloc((size_t)(g.ssm_d_conv - 1) * g.ssm_conv_channels));
            s.S.push_back(dalloc((size_t)g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size));
            s.kc.push_back({});
            s.vc.push_back({});
        }
    }
}

std::string L(int64_t l, const char *s) { return "blk." + std::to_string(l) + "." + s; }

void embed(const Model &m, State &s, int tok) {
    const Tensor &e = m.W("token_embd.weight");
    dispatch("q36_row_" + std::to_string(e.type), 4, 128, e.d, (uint64_t)e.row_bytes, tok, (int)m.g.n_embd, s.x);
}

/// One token through the model at position s.pos; the layer outputs into `layers` (host, n_layers x N) when given.
void forward(const Model &m, State &s, int tok, float *layers = nullptr) {
    const auto &g = m.g;
    const int N = (int)g.n_embd, K = (int)m.top_k, FF = (int)g.n_ff, FS = (int)g.n_ff_shexp;
    const int B = 256;
    embed(m, s, tok);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        dispatch("q36_rms", 1, kThreads, s.x, m.F(L(l, "attn_norm.weight")), s.h, N, m.eps);
        if (!strata::core::is_qsa_layer(g, l)) {
            const int HK = (int)g.ssm_k_heads, HV = (int)g.ssm_v_heads, D = (int)g.ssm_state_size;
            const int C = (int)g.ssm_conv_channels;
            gemv(m.W(L(l, "attn_qkv.weight")), s.h, s.qkv);
            gemv(m.W(L(l, "attn_gate.weight")), s.h, s.z);
            gemv(m.W(L(l, "ssm_beta.weight")), s.h, s.beta);
            gemv(m.W(L(l, "ssm_alpha.weight")), s.h, s.alpha);
            dispatch("q36_conv", (C + B - 1) / B, B, s.qkv, s.conv_state[l], m.F(L(l, "ssm_conv1d.weight")), s.conv_out,
                     C, (int)g.ssm_d_conv);
            dispatch("q36_gdn", HV, D, s.conv_out, s.alpha, m.F(L(l, "ssm_dt.bias")), m.F(L(l, "ssm_a")), s.beta,
                     s.S[l], s.gdn_out, HK, HV, D, m.eps);
            dispatch("q36_gated_norm", HV, D, s.gdn_out, m.F(L(l, "ssm_norm.weight")), s.z, D, m.eps);
            gemv(m.W(L(l, "ssm_out.weight")), s.gdn_out, s.y);
        } else {
            const int H = (int)g.n_head, HKV = (int)g.n_head_kv, D = (int)g.head_dim;
            gemv(m.W(L(l, "attn_q.weight")), s.h, s.qg);
            gemv(m.W(L(l, "attn_k.weight")), s.h, s.k);
            gemv(m.W(L(l, "attn_v.weight")), s.h, s.v);
            dispatch("q36_attn_prep", H + HKV, kThreads, s.qg, s.k, s.v, m.F(L(l, "attn_q_norm.weight")),
                     m.F(L(l, "attn_k_norm.weight")), s.q, s.gate, s.kc[l], s.vc[l], H, HKV, D, m.n_rot, s.pos, s.ctx,
                     m.rope_base, m.eps);
            dispatch("q36_attn", H, kThreads, s.q, s.kc[l], s.vc[l], s.gate, s.att, H, HKV, D, s.pos + 1, s.ctx,
                     1.0f / sqrtf((float)D));
            gemv(m.W(L(l, "attn_output.weight")), s.att, s.y);
        }

        dispatch("q36_add", (N + B - 1) / B, B, s.x, s.y, N);
        dispatch("q36_rms", 1, kThreads, s.x, m.F(L(l, "post_attention_norm.weight")), s.h, N, m.eps);
        // router: softmax over every expert, top-k, renormalised (host side: exact and simple)
        gemv(m.W(L(l, "ffn_gate_inp.weight")), s.h, s.router);
        std::vector<float> r((size_t)g.n_expert);
        flush();
        std::memcpy(r.data(), s.router.ptr, r.size() * 4);
        const float mx = *std::max_element(r.begin(), r.end());
        double sum = 0;
        for (float &v : r)
            sum += (v = std::exp(v - mx));
        for (float &v : r)
            v = (float)(v / sum);
        std::vector<int> idx((size_t)g.n_expert);
        for (int i = 0; i < (int)idx.size(); ++i)
            idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                          [&](int a, int b) { return r[a] > r[b] || (r[a] == r[b] && a < b); });
        std::vector<float> w(K);
        float ws = 0;
        for (int i = 0; i < K; ++i)
            ws += (w[i] = r[idx[i]]);
        for (float &v : w)
            v /= ws;
        g_ids_host.assign(idx.begin(), idx.begin() + K);
        std::memcpy(s.ids.ptr, idx.data(), K * sizeof(int));
        std::memcpy(s.wts.ptr, w.data(), K * 4);
        gemv(m.W(L(l, "ffn_gate_exps.weight")), s.h, s.eg, &s.ids, K, 0);
        gemv(m.W(L(l, "ffn_up_exps.weight")), s.h, s.eu, &s.ids, K, 0);
        dispatch("q36_swiglu", (K * FF + B - 1) / B, B, s.eg, s.eu, s.eh, K * FF);
        gemv(m.W(L(l, "ffn_down_exps.weight")), s.eh, s.ey, &s.ids, K, FF);
        // shared expert with its sigmoid scalar gate
        gemv(m.W(L(l, "ffn_gate_shexp.weight")), s.h, s.sg);
        gemv(m.W(L(l, "ffn_up_shexp.weight")), s.h, s.su);
        dispatch("q36_swiglu", (FS + B - 1) / B, B, s.sg, s.su, s.sh, FS);
        gemv(m.W(L(l, "ffn_down_shexp.weight")), s.sh, s.sy);
        {
            Tensor gs;
            gs.d = m.F(L(l, "ffn_gate_inp_shexp.weight"));
            gs.type = T_F32;
            gs.shape = {N, 1};
            gs.row_bytes = (size_t)N * 4;
            gemv(gs, s.h, s.sgate);
            flush(); // dispatch bindings refer to this temporary tensor's buffer descriptor
        }
        dispatch("q36_moe_sum", (N + B - 1) / B, B, s.x, s.ey, s.wts, K, s.sy, s.sgate, N);

        flush();
        if (layers) {
            std::memcpy(layers + (size_t)l * N, s.x.ptr, N * 4);
        }
    }
    dispatch("q36_rms", 1, kThreads, s.x, m.F("output_norm.weight"), s.h, N, m.eps);
    gemv(m.W("output.weight"), s.h, s.logits);
    flush();
    ++s.pos;
}

struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

int sample(std::vector<float> &lg, const Sampling &s, const std::vector<int> &hist, std::mt19937_64 &rng) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t)s.last_n >= hist.size() ? 0 : hist.size() - (size_t)s.last_n;
        for (size_t i = from; i < hist.size(); ++i)
            ++cnt[hist[i]];
        for (const auto &[id, c] : cnt) {
            float &v = lg[(size_t)id];
            if (s.repeat != 1.f)
                v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float)c * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f)
        return (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int)lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i)
        idx[i] = (int)i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t)kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t)idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i)
        sum += (p[i] = std::exp((lg[(size_t)idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) {
            keep = i;
            break;
        }
        cum += p[i] / sum;
        if (cum >= s.top_p) {
            keep = i + 1;
            break;
        }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i)
        tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i) {
        if ((r -= p[i]) <= 0)
            return idx[i];
    }
    return idx[keep - 1];
}

std::vector<int> parse_ids(const char *s) {
    std::vector<int> v;
    while (*s) {
        while (*s == ',' || *s == ' ')
            ++s;
        if (!*s)
            break;
        char *e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s)
            break;
        v.push_back((int32_t)x);
        s = e;
    }
    return v;
}

bool is_eos(int t) { return t == 248044 || t == 248046; }

void reset(State &s) {
    flush();
    s.pos = 0;
    for (const auto *list : {&s.conv_state, &s.S, &s.kc, &s.vc})
        for (const auto &b : *list)
            if (b.ptr)
                std::memset(b.ptr, 0, b.bytes);
}
int serve(const Model &m, State &s) {
    std::printf("READY %d stop\n", s.ctx);
    std::fflush(stdout);
    struct Inbox {
        std::mutex mu;
        std::condition_variable cv;
        std::deque<std::string> lines;
        bool eof = false;
        std::atomic<bool> stop{false};
    };
    auto inbox = std::make_shared<Inbox>();
    std::thread reader([inbox] {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            if (line == "STOP") {
                inbox->stop.store(true);
                continue;
            }
            std::lock_guard<std::mutex> lock(inbox->mu);
            inbox->lines.push_back(line);
            inbox->cv.notify_one();
        }
        std::lock_guard<std::mutex> lock(inbox->mu);
        inbox->eof = true;
        inbox->cv.notify_one();
    });
    reader.detach();
    std::vector<int> live;
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lock(inbox->mu);
            inbox->cv.wait(lock, [&] { return inbox->eof || !inbox->lines.empty(); });
            if (inbox->lines.empty())
                return 0;
            line = inbox->lines.front();
            inbox->lines.pop_front();
        }
        if (line.empty())
            continue;
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected GEN\n");
            std::fflush(stdout);
            continue;
        }
        char *p = nullptr;
        long long count = std::strtoll(line.c_str() + 4, &p, 10);
        Sampling smp;
        for (;;) {
            while (*p == ' ')
                ++p;
            const char *begin = p;
            while (*p && *p != ' ')
                ++p;
            std::string part(begin, p - begin);
            auto eq = part.find('=');
            if (part.empty() || eq == std::string::npos) {
                p = const_cast<char *>(begin);
                break;
            }
            auto key = part.substr(0, eq);
            const char *v = part.c_str() + eq + 1;
            if (key == "temperature")
                smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p")
                smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k")
                smp.top_k = std::atoi(v);
            else if (key == "min_p")
                smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n")
                smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat")
                smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq")
                smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present")
                smp.present = std::strtof(v, nullptr);
            else if (key == "seed")
                smp.seed = std::strtoull(v, nullptr, 10);
        }
        auto ids = parse_ids(p);
        if (count < 1 || ids.empty() || count > s.ctx || ids.size() > size_t(s.ctx - count)) {
            std::printf("ERR prompt + max_new exceeds context or is empty\n");
            std::fflush(stdout);
            continue;
        }
        if (std::any_of(ids.begin(), ids.end(), [&](int t) { return t < 0 || t >= m.g.n_vocab; })) {
            std::printf("ERR token outside vocabulary\n");
            std::fflush(stdout);
            continue;
        }
        size_t reuse = 0;
        if (live.size() < ids.size() && std::equal(live.begin(), live.end(), ids.begin()))
            reuse = live.size();
        else {
            reset(s);
            live.clear();
        }
        inbox->stop.store(false);
        const double t0 = mtl::now_s();
        size_t fresh = ids.size() - reuse, done = 0;
        for (size_t i = reuse; i < ids.size(); ++i) {
            if (inbox->stop.load())
                break;
            forward(m, s, ids[i]);
            live.push_back(ids[i]);
            ++done;
            const double ms = (mtl::now_s() - t0) * 1000;
            std::printf("PP %zu %zu %.1f %.1f\n", done, fresh, ms, done / std::max(1e-9, ms / 1000));
            std::fflush(stdout);
        }
        const double pp_ms = (mtl::now_s() - t0) * 1000, d0 = mtl::now_s();
        long long produced = 0;
        const char *finish = "length";
        auto hist = ids;
        std::mt19937_64 rng(smp.seed);
        std::vector<float> logits(m.g.n_vocab);
        while (produced < count) {
            if (inbox->stop.load()) {
                finish = "cancel";
                break;
            }
            std::memcpy(logits.data(), s.logits.ptr, logits.size() * 4);
            int next = sample(logits, smp, hist, rng);
            std::printf("T %d\n", next);
            std::fflush(stdout);
            ++produced;
            hist.push_back(next);
            if (is_eos(next)) {
                finish = "stop";
                break;
            }
            if (produced == count)
                break;
            forward(m, s, next);
            live.push_back(next);
        }
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %zu 0\n", produced, ids.size(), pp_ms,
                    (mtl::now_s() - d0) * 1000, finish, reuse, done);
        std::fflush(stdout);
    }
}
} // namespace

int main(int argc, char **argv) {
    std::string model, tokens, force, dump;
    int ctx = 4096, n = 64;
    bool serving = false, kernels_only = false;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc)
                    throw std::runtime_error(a + " needs a value");
                return argv[++i];
            };
            if (a == "--native" || a == "--model")
                model = next();
            else if (a == "--max-context" || a == "--ctx")
                ctx = std::stoi(next());
            else if (a == "--tokens")
                tokens = next();
            else if (a == "--n")
                n = std::stoi(next());
            else if (a == "--force")
                force = next();
            else if (a == "--dump-full-logits")
                dump = next();
            else if (a == "--serve")
                serving = true;
            else if (a == "--check-kernels")
                kernels_only = true;
            else if (a == "--version") {
                std::puts("strata-q36-metal experimental");
                return 0;
            } else if (a == "--help") {
                std::puts("strata-q36-metal --native MODEL --max-context 4096 --serve\n  --tokens IDS --n N [--force "
                          "IDS --dump-full-logits FILE]\n  --check-kernels");
                return 0;
            } else {
                std::fprintf(stderr, "strata-q36-metal: unknown option %s\n", a.c_str());
                return 2;
            }
        }
        if (ctx < 1 || ctx > 4096)
            throw std::runtime_error("experimental Metal Qwen3.6 context must be 1..4096");
        lib = cx.compile({"q36.metal"}, "#define NO_GGML_COMMON\n");
        if (kernels_only) {
            for (int t : {0, 1, 8, 12, 13, 14, 30}) {
                cx.pipeline(lib, "q36_gemv_" + std::to_string(t));
                cx.pipeline(lib, "q36_row_" + std::to_string(t));
            }
            for (auto name : {"q36_rms", "q36_conv", "q36_gdn", "q36_gated_norm", "q36_attn_prep", "q36_attn",
                              "q36_swiglu", "q36_add", "q36_moe_sum"})
                cx.pipeline(lib, name);
            std::puts("Metal Qwen3.6 kernels compiled");
            return 0;
        }
        if (model.empty())
            throw std::runtime_error("--native MODEL is required");
        Model m;
        std::string error;
        if (!load(model, m, error))
            throw std::runtime_error(error);
        State s;
        s.ctx = ctx;
        init_state(m, s);
        std::fprintf(stderr, "strata-q36-metal: %s, %.2f GiB buffers, selected experts streamed; MTP and vision off\n",
                     cx.device_name().c_str(), cx.allocated() / 1073741824.0);
        if (serving)
            return serve(m, s);
        auto prompt = parse_ids(tokens.c_str()), forced = parse_ids(force.c_str());
        if (prompt.empty() || n < 1 || prompt.size() + size_t(n) > size_t(ctx))
            throw std::runtime_error("invalid prompt/generation/context");
        if (std::any_of(prompt.begin(), prompt.end(), [&](int t) { return t < 0 || t >= m.g.n_vocab; }) ||
            std::any_of(forced.begin(), forced.end(), [&](int t) { return t < 0 || t >= m.g.n_vocab; }))
            throw std::runtime_error("token outside vocabulary");
        auto t0 = mtl::now_s();
        for (int t : prompt)
            forward(m, s, t);
        auto d0 = mtl::now_s();
        std::vector<int> generated;
        FILE *f = dump.empty() ? nullptr : std::fopen(dump.c_str(), "wb");
        if (!dump.empty() && !f)
            throw std::runtime_error("cannot open logits output");
        for (int i = 0; i < n; ++i) {
            auto lg = s.logits.as<float>();
            int t = std::max_element(lg, lg + m.g.n_vocab) - lg;
            generated.push_back(t);
            if (f)
                std::fwrite(lg, 4, m.g.n_vocab, f);
            if (i + 1 < n)
                forward(m, s, i < int(forced.size()) ? forced[i] : t);
        }
        if (f)
            std::fclose(f);
        std::printf("gen:");
        for (int t : generated)
            std::printf(" %d", t);
        std::puts("");
        std::fprintf(stderr, "strata-q36-metal: pp %zu in %.2fs; tg %d in %.2fs (%.2f tok/s)\n", prompt.size(), d0 - t0,
                     n, mtl::now_s() - d0, n / std::max(1e-9, mtl::now_s() - d0));
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "strata-q36-metal: %s\n", e.what());
        return 1;
    }
}
