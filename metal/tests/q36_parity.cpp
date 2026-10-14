// Numerical tests for the experimental Qwen3.6 decode kernels, independent of a model download.
#define main q36_engine_entry
#include "../src/q36_engine.cpp"
#undef main
#include "ref_quant.h"

static int failures = 0;
static void check(const char *name, const float *actual, const std::vector<float> &expected, double limit = 1e-5) {
    double diff = 0, norm = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        double e = actual[i] - expected[i];
        diff += e * e;
        norm += double(expected[i]) * expected[i];
    }
    double rel = std::sqrt(diff / std::max(1e-30, norm));
    bool ok = std::isfinite(rel) && rel < limit;
    std::printf("%-28s relative L2 %.3e %s\n", name, rel, ok ? "PASS" : "FAIL");
    failures += !ok;
}
static void matmul_test() {
    for (int type : {0, 1, 8, 12, 13, 14, 30}) {
        constexpr int ni = 256, no = 32, k = 3;
        Tensor w;
        w.type = type;
        w.shape = {ni, no};
        w.row_bytes = row_bytes_of(type, ni);
        w.d = cx.alloc(w.row_bytes * no);
        ref::random_blocks(type, w.d.as<uint8_t>(), no, ni, 42);
        auto x = cx.alloc(ni * 4), y = cx.alloc(no * 4), row = cx.alloc(ni * 4);
        ref::Rng random(17);
        for (int i = 0; i < ni; ++i)
            x.as<float>()[i] = random.sym();
        std::vector<float> weights(ni * no), expected(no), erow(ni);
        ref::dequant(type, w.d.ptr, weights.data(), weights.size());
        for (int j = 0; j < no; ++j) {
            double sum = 0;
            for (int i = 0; i < ni; ++i)
                sum += double(weights[j * ni + i]) * x.as<float>()[i];
            expected[j] = sum;
        }
        gemv(w, x, y);
        dispatch("q36_row_" + std::to_string(type), 4, 128, w.d, uint64_t(w.row_bytes), 7, ni, row);
        flush();
        check((std::string("gemv ") + ref::type_name(type)).c_str(), y.as<float>(), expected);
        std::copy(weights.begin() + 7 * ni, weights.begin() + 8 * ni, erow.begin());
        check("embedding row", row.as<float>(), erow);
        // Selected experts with distinct inputs; both host staging and grid.y must select the right matrix.
        Tensor experts;
        experts.type = type;
        experts.shape = {ni, no, 5};
        experts.row_bytes = w.row_bytes;
        std::vector<uint8_t> source(w.row_bytes * no * 5);
        ref::random_blocks(type, source.data(), no * 5, ni, 71);
        experts.source = source.data();
        experts.d = cx.alloc(w.row_bytes * no * k);
        auto xx = cx.alloc(ni * k * 4), yy = cx.alloc(no * k * 4), ids = cx.alloc(k * 4);
        for (int i = 0; i < ni * k; ++i)
            xx.as<float>()[i] = random.sym();
        g_ids_host = {4, 0, 2};
        std::vector<float> ew(ni * no * 5), ey(no * k);
        ref::dequant(type, source.data(), ew.data(), ew.size());
        for (int e = 0; e < k; ++e)
            for (int j = 0; j < no; ++j) {
                double sum = 0;
                for (int i = 0; i < ni; ++i)
                    sum += double(ew[(g_ids_host[e] * no + j) * ni + i]) * xx.as<float>()[e * ni + i];
                ey[e * no + j] = sum;
            }
        gemv(experts, xx, yy, &ids, k, ni);
        flush();
        check("selected expert gemv", yy.as<float>(), ey);
        for (auto b : {w.d, x, y, row, experts.d, xx, yy, ids})
            cx.free(b);
    }
}
static void gdn_test() {
    constexpr int hk = 2, hv = 4, d = 128, c = 2 * hk * d + hv * d, taps = 4;
    ref::Rng r(22);
    auto qkv = dalloc(c), conv = dalloc(c), history = dalloc((taps - 1) * c), w = dalloc(c * taps);
    auto alpha = dalloc(hv), beta = dalloc(hv), dt = dalloc(hv), a = dalloc(hv), state = dalloc(hv * d * d),
         out = dalloc(hv * d), norm = dalloc(d), z = dalloc(hv * d);
    for (int i = 0; i < c * taps; ++i)
        w.as<float>()[i] = r.sym() * .1f;
    for (int i = 0; i < hv; ++i) {
        alpha.as<float>()[i] = r.sym();
        beta.as<float>()[i] = r.sym();
        dt.as<float>()[i] = r.sym();
        a.as<float>()[i] = -.5f;
    }
    for (int i = 0; i < d; ++i)
        norm.as<float>()[i] = 1 + r.sym() * .1f;
    std::vector<float> hs((taps - 1) * c, 0), st(hv * d * d, 0), conv_ref(c), raw(hv * d), final(hv * d);
    for (int step = 0; step < 3; ++step) {
        for (int i = 0; i < c; ++i)
            qkv.as<float>()[i] = r.sym();
        for (int i = 0; i < hv * d; ++i)
            z.as<float>()[i] = r.sym();
        for (int i = 0; i < c; ++i) {
            float acc = 0;
            for (int t = 0; t < taps - 1; ++t)
                acc += hs[t * c + i] * w.as<float>()[i * taps + t];
            acc += qkv.as<float>()[i] * w.as<float>()[i * taps + taps - 1];
            for (int t = 0; t < taps - 2; ++t)
                hs[t * c + i] = hs[(t + 1) * c + i];
            hs[(taps - 2) * c + i] = qkv.as<float>()[i];
            conv_ref[i] = acc / (1 + std::exp(-acc));
        }
        for (int h = 0; h < hv; ++h) {
            int kh = h % hk;
            double qn = 1e-6, kn = 1e-6;
            for (int j = 0; j < d; ++j) {
                qn += double(conv_ref[kh * d + j]) * conv_ref[kh * d + j];
                kn += double(conv_ref[hk * d + kh * d + j]) * conv_ref[hk * d + kh * d + j];
            }
            float val = alpha.as<float>()[h] + dt.as<float>()[h];
            float decay = std::exp(std::log1p(std::exp(val)) * a.as<float>()[h]);
            float b = 1 / (1 + std::exp(-beta.as<float>()[h]));
            for (int i = 0; i < d; ++i) {
                float sk = 0;
                for (int j = 0; j < d; ++j) {
                    st[(h * d + i) * d + j] *= decay;
                    sk += st[(h * d + i) * d + j] * conv_ref[hk * d + kh * d + j] / std::sqrt(kn);
                }
                float delta = (conv_ref[2 * hk * d + h * d + i] - sk) * b;
                float acc = 0;
                for (int j = 0; j < d; ++j) {
                    st[(h * d + i) * d + j] += delta * conv_ref[hk * d + kh * d + j] / std::sqrt(kn);
                    acc += st[(h * d + i) * d + j] * conv_ref[kh * d + j] / std::sqrt(qn * d);
                }
                raw[h * d + i] = acc;
            }
            double ss = 0;
            for (int i = 0; i < d; ++i)
                ss += double(raw[h * d + i]) * raw[h * d + i];
            for (int i = 0; i < d; ++i) {
                float zz = z.as<float>()[h * d + i];
                final[h * d + i] =
                    raw[h * d + i] / std::sqrt(ss / d + 1e-6) * norm.as<float>()[i] * zz / (1 + std::exp(-zz));
            }
        }
        dispatch("q36_conv", (c + 255) / 256, 256, qkv, history, w, conv, c, taps);
        dispatch("q36_gdn", hv, d, conv, alpha, dt, a, beta, state, out, hk, hv, d, 1e-6f);
        flush();
        check("GDN recurrence", out.as<float>(), raw);
        dispatch("q36_gated_norm", hv, d, out, norm, z, d, 1e-6f);
        flush();
        check("GDN SiLU gated norm", out.as<float>(), final);
        check("GDN state", state.as<float>(), st);
    }
}
static void attention_test() {
    constexpr int h = 4, hkv = 2, d = 256, ctx = 64, pos = 3, rot = 64;
    ref::Rng r(33);
    auto qg = dalloc(2 * h * d), kx = dalloc(hkv * d), vx = dalloc(hkv * d), qw = dalloc(d), kw = dalloc(d),
         q = dalloc(h * d), gate = dalloc(h * d), kc = dalloc(hkv * ctx * d), vc = dalloc(hkv * ctx * d),
         out = dalloc(h * d);
    for (auto b : {qg, kx, vx, kc, vc})
        for (size_t i = 0; i < b.bytes / 4; ++i)
            b.as<float>()[i] = r.sym();
    for (int i = 0; i < d; ++i) {
        qw.as<float>()[i] = 1 + r.sym() * .1f;
        kw.as<float>()[i] = 1 + r.sym() * .1f;
    }
    std::vector<float> qr(h * d), kr(hkv * d), gr(h * d), expected(h * d);
    for (int head = 0; head < h + hkv; ++head) {
        bool query = head < h;
        const float *src = query ? qg.as<float>() + head * 2 * d : kx.as<float>() + (head - h) * d;
        double ss = 0;
        for (int i = 0; i < d; ++i)
            ss += double(src[i]) * src[i];
        std::vector<float> normed(d);
        for (int i = 0; i < d; ++i)
            normed[i] = src[i] / std::sqrt(ss / d + 1e-6) * (query ? qw.as<float>()[i] : kw.as<float>()[i]);
        for (int i = 0; i < d; ++i) {
            float v = normed[i];
            if (i < rot) {
                int p = i < rot / 2 ? i : i - rot / 2;
                float angle = pos * std::pow(1e7f, -2.f * p / rot);
                v = i < rot / 2 ? normed[i] * std::cos(angle) - normed[i + rot / 2] * std::sin(angle)
                                : normed[i - rot / 2] * std::sin(angle) + normed[i] * std::cos(angle);
            }
            if (query) {
                qr[head * d + i] = v;
                gr[head * d + i] = qg.as<float>()[head * 2 * d + d + i];
            } else
                kr[(head - h) * d + i] = v;
        }
    }
    dispatch("q36_attn_prep", h + hkv, 128, qg, kx, vx, qw, kw, q, gate, kc, vc, h, hkv, d, rot, pos, ctx, 1e7f, 1e-6f);
    flush();
    check("attention Q norm and RoPE", q.as<float>(), qr);
    for (int head = 0; head < h; ++head) {
        int kv = head / (h / hkv);
        std::vector<double> scores(pos + 1);
        double mx = -1e30, sum = 0;
        for (int p = 0; p <= pos; ++p) {
            double dot = 0;
            for (int i = 0; i < d; ++i)
                dot += double(qr[head * d + i]) * (p == pos ? kr[kv * d + i] : kc.as<float>()[(kv * ctx + p) * d + i]);
            scores[p] = dot / std::sqrt(d);
            mx = std::max(mx, scores[p]);
        }
        for (auto &v : scores) {
            v = std::exp(v - mx);
            sum += v;
        }
        for (int i = 0; i < d; ++i) {
            double v = 0;
            for (int p = 0; p <= pos; ++p)
                v += scores[p] * vc.as<float>()[(kv * ctx + p) * d + i];
            expected[head * d + i] = v / sum / (1 + std::exp(-gr[head * d + i]));
        }
    }
    dispatch("q36_attn", h, 128, q, kc, vc, gate, out, h, hkv, d, pos + 1, ctx, 1.f / std::sqrt(float(d)));
    flush();
    check("gated grouped attention", out.as<float>(), expected);
}
int main() {
    lib = cx.compile({"q36.metal"}, "#define NO_GGML_COMMON\n");
    matmul_test();
    gdn_test();
    attention_test();
    return failures ? 1 : 0;
}
