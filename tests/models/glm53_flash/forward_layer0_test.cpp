// Forward harness for the three leading dense KDA layers of GLM-5.3-Flash.
// Runs real weights from the GGUF and dumps per-stage activations for an
// independent reimplementation to compare against (see layer0_numpy.py).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/runtime/config.hpp"
#include "src/models/glm53_flash/runtime/manifest.hpp"

namespace glm = gufo::models::glm53_flash;
using gufo::core::GgmlType;
using gufo::core::GgufReader;
using gufo::core::GgufTensorInfo;

namespace {

std::unordered_map<std::string, const GgufTensorInfo*> g_tensors;
std::string g_stage_prefix;

float Sigmoid(float x) {
  return 1.0F / (1.0F + std::exp(-x));
}

void Rms(std::span<float> x, const float* w, float eps) {
  double sum = 0.0;
  for (const auto v : x)
    sum += static_cast<double>(v) * v;
  const float inv = static_cast<float>(
      1.0 / std::sqrt(sum / static_cast<double>(x.size()) + eps));
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = x[i] * inv * (w != nullptr ? w[i] : 1.0F);
  }
}

void L2(std::span<float> x, float eps) {
  double sum = 0.0;
  for (const auto v : x)
    sum += static_cast<double>(v) * v;
  const float inv = static_cast<float>(1.0 / std::sqrt(sum + eps));
  for (auto& v : x)
    v *= inv;
}

// LLM_NORM: rms without eps on the mean? llama.cpp: x*w/sqrt(mean(x^2)+eps)+b.
void Lnorm(std::span<float> x, const float* w, const float* b, float eps) {
  double sum = 0.0;
  for (const auto v : x)
    sum += static_cast<double>(v) * v;
  const float inv = static_cast<float>(
      1.0 / std::sqrt(sum / static_cast<double>(x.size()) + eps));
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = x[i] * inv * (w != nullptr ? w[i] : 1.0F) +
           (b != nullptr ? b[i] : 0.0F);
  }
}

// Full dequantization of a tensor to F32, row-major over the last dimension.
class F32 {
public:
  F32() = default;
  explicit F32(const GgufTensorInfo* t) {
    Setup(t);
    data_.resize(static_cast<std::size_t>(rows_) * cols_);
    for (std::uint64_t r = 0; r < rows_; ++r) {
      DequantizeRow(r, data_.data() + r * cols_);
    }
  }
  // View that only materializes single rows; used for the embedding table.
  static F32 RowView(const GgufTensorInfo* t) {
    F32 f;
    f.Setup(t);
    return f;
  }

  void Setup(const GgufTensorInfo* t) {
    t_ = t;
    rows_ = 1;
    for (std::size_t d = 1; d < t->dimensions.size(); ++d) {
      rows_ *= t->dimensions[d];
    }
    cols_ = t->dimensions[0];
  }

  void DequantizeRow(std::uint64_t row, float* out) const {
    const std::size_t rb = static_cast<std::size_t>(
        t_->size_bytes / std::max<std::uint64_t>(rows_, 1));
    const char* src = static_cast<const char*>(t_->data) + row * rb;
    switch (t_->type) {
      case GgmlType::kF32:
        std::memcpy(out, src, cols_ * sizeof(float));
        break;
      case GgmlType::kQ5_K:
        gufo::quant::DequantizeQ5_K(src, out, cols_);
        break;
      case GgmlType::kQ6_K:
        gufo::quant::DequantizeQ6_K(src, out, cols_);
        break;
      case GgmlType::kQ8_0:
        gufo::quant::DequantizeQ8_0(src, out, cols_);
        break;
      case GgmlType::kIQ2_XXS:
        gufo::quant::DequantizeIQ2_XXS(src, out, cols_);
        break;
      case GgmlType::kIQ2_S:
        gufo::quant::DequantizeIQ2_S(src, out, cols_);
        break;
      default:
        std::fprintf(stderr, "unsupported type %s\n",
                     std::string(gufo::core::ToString(t_->type)).c_str());
        std::abort();
    }
  }

  void MatVec(std::span<const float> x, std::span<float> y) const {
    for (std::uint64_t r = 0; r < rows_; ++r) {
      const float* w = data_.data() + r * cols_;
      double acc = 0.0;
      for (std::uint64_t i = 0; i < cols_; ++i) {
        acc += static_cast<double>(w[i]) * x[i];
      }
      y[r] = static_cast<float>(acc);
    }
  }

  void MatVecRows(std::uint64_t row0, std::uint64_t nrows,
                  std::span<const float> x, std::span<float> y) const {
    for (std::uint64_t r = 0; r < nrows; ++r) {
      const float* w = data_.data() + (row0 + r) * cols_;
      double acc = 0.0;
      for (std::uint64_t i = 0; i < cols_; ++i) {
        acc += static_cast<double>(w[i]) * x[i];
      }
      y[r] = static_cast<float>(acc);
    }
  }

  // Row-by-row dequantizing matvec for RowView instances.
  void MatVecView(std::uint64_t row0, std::uint64_t nrows,
                  std::span<const float> x, std::span<float> y) const {
    std::vector<float> rowbuf(cols_);
    for (std::uint64_t r = 0; r < nrows; ++r) {
      DequantizeRow(row0 + r, rowbuf.data());
      double acc = 0.0;
      for (std::uint64_t i = 0; i < cols_; ++i) {
        acc += static_cast<double>(rowbuf[i]) * x[i];
      }
      y[r] = static_cast<float>(acc);
    }
  }

  std::uint64_t rows() const { return rows_; }
  std::uint64_t cols() const { return cols_; }
  const std::vector<float>& data() const { return data_; }

private:
  const GgufTensorInfo* t_{nullptr};
  std::uint64_t rows_{0}, cols_{0};
  std::vector<float> data_;
};

const F32& Load(const char* name) {
  static std::unordered_map<std::string, F32> cache;
  auto it = cache.find(name);
  if (it != cache.end()) {
    return it->second;
  }
  const auto found = g_tensors.find(name);
  if (found == g_tensors.end()) {
    std::fprintf(stderr, "missing tensor %s\n", name);
    std::abort();
  }
  return cache.emplace(name, F32(found->second)).first->second;
}

// Row-view cache for tensors too large to materialize fully (MoE experts).
const F32& LoadRow(const char* name) {
  static std::unordered_map<std::string, F32> cache;
  auto it = cache.find(name);
  if (it != cache.end()) {
    return it->second;
  }
  const auto found = g_tensors.find(name);
  if (found == g_tensors.end()) {
    std::fprintf(stderr, "missing tensor %s\n", name);
    std::abort();
  }
  return cache.emplace(name, F32::RowView(found->second)).first->second;
}

void Dump(const char* dir, const char* stage, std::uint32_t token,
          std::span<const float> v) {
  if (dir == nullptr) {
    return;
  }
  std::string path(dir);
  path += "/";
  path += g_stage_prefix;
  path += stage;
  path += ".";
  path += std::to_string(token);
  path += ".f32";
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(float)));
}

struct HcMix {
  std::vector<float> pre, post, comb;
};

HcMix HcPre(std::span<const float> x, const char* prefix, float eps, int token,
            const char* dump_dir) {
  const std::uint32_t hc = 4, h = 4096;
  std::vector<float> xn(x.begin(), x.end());
  Rms(xn, nullptr, 1e-5F);
  std::vector<float> mixes(24);
  Load((std::string(prefix) + "_fn.weight").c_str()).MatVec(xn, mixes);
  const float* scale =
      Load((std::string(prefix) + "_scale.weight").c_str()).data().data();
  const float* base =
      Load((std::string(prefix) + "_base.weight").c_str()).data().data();
  HcMix m;
  m.pre.resize(hc);
  m.post.resize(hc);
  m.comb.resize(hc * hc);
  for (std::uint32_t s = 0; s < hc; ++s) {
    m.pre[s] = Sigmoid(mixes[s] * scale[0] + base[s]) + eps;
    m.post[s] = Sigmoid(mixes[hc + s] * scale[1] + base[hc + s]) * 2.0F;
  }
  for (std::uint32_t i = 0; i < hc * hc; ++i) {
    m.comb[i] = mixes[2 * hc + i] * scale[2] + base[2 * hc + i];
  }
  // comb[dst + hc*src]; softmax over dst, then sinkhorn.
  for (std::uint32_t src = 0; src < hc; ++src) {
    double mx = -1e30, sum = 0.0;
    for (std::uint32_t d = 0; d < hc; ++d) {
      mx = std::max(mx, static_cast<double>(m.comb[d + hc * src]));
    }
    for (std::uint32_t d = 0; d < hc; ++d) {
      m.comb[d + hc * src] =
          static_cast<float>(std::exp(m.comb[d + hc * src] - mx));
      sum += m.comb[d + hc * src];
    }
    for (std::uint32_t d = 0; d < hc; ++d) {
      m.comb[d + hc * src] /= static_cast<float>(sum + eps);
    }
  }
  const auto norm = [&](bool over_src) {
    for (std::uint32_t a = 0; a < hc; ++a) {
      double sum = 0.0;
      for (std::uint32_t b = 0; b < hc; ++b) {
        sum += over_src ? m.comb[a + hc * b] : m.comb[b + hc * a];
      }
      const float denom = static_cast<float>(sum + eps);
      for (std::uint32_t b = 0; b < hc; ++b) {
        if (over_src) {
          m.comb[a + hc * b] /= denom;
        } else {
          m.comb[b + hc * a] /= denom;
        }
      }
    }
  };
  norm(true);
  for (int i = 0; i < 19; ++i) {
    norm(false);
    norm(true);
  }
  Dump(dump_dir, prefix, token, mixes);
  Dump((std::string(dump_dir) + "/pre").c_str(), prefix, token, m.pre);
  Dump((std::string(dump_dir) + "/comb").c_str(), prefix, token, m.comb);
  Dump((std::string(dump_dir) + "/post").c_str(), prefix, token, m.post);
  return m;
}

void HcPost(std::span<const float> block_out, std::span<const float> res,
            const HcMix& m, std::vector<float>& out) {
  const std::uint32_t hc = 4, h = 4096;
  out.assign(res.size(), 0.0F);
  for (std::uint32_t d = 0; d < hc; ++d) {
    for (std::uint32_t i = 0; i < h; ++i) {
      float acc = block_out[i] * m.post[d];
      for (std::uint32_t s = 0; s < hc; ++s) {
        acc += res[s * h + i] * m.comb[d + hc * s];
      }
      out[d * h + i] = acc;
    }
  }
}

// ---- DSA / MLA (nope-only, kpool-select indexer) ----
constexpr std::uint32_t kKv = 512, kQk = 256, kNh = 64, kId = 128, kIh = 32;
constexpr std::uint32_t kKpool = 4, kTopPool = 2048 / kKpool;

struct DsaState {
  std::vector<float> lat;     // n * kKv
  std::vector<float> ik;      // n * kId (normalized)
  std::vector<float> gate;    // n * kId
  std::vector<float> pooled;  // npool * kId
};

void DsaFwd(std::uint32_t layer, std::uint32_t t, std::span<const float> xn,
            std::span<float> block_out, DsaState& st, const char* dump_dir) {
  const auto p = [layer](const char* s) {
    return "blk." + std::to_string(layer) + "." + s;
  };
  std::vector<float> qr(1536), q(kNh * kQk), lat(kKv);
  Load(p("attn_q_a.weight").c_str()).MatVec(xn, qr);
  Rms(qr, Load(p("attn_q_a_norm.weight").c_str()).data().data(), 1e-5F);
  Load(p("attn_q_b.weight").c_str()).MatVec(qr, q);
  Load(p("attn_kv_a_mqa.weight").c_str()).MatVec(xn, lat);
  Rms(lat, Load(p("attn_kv_a_norm.weight").c_str()).data().data(), 1e-5F);

  std::vector<float> iq(kIh * kId);
  Load(p("indexer.attn_q_b.weight").c_str()).MatVec(qr, iq);
  std::vector<float> ik(kId);
  Load(p("indexer.attn_k.weight").c_str()).MatVec(xn, ik);
  Lnorm(ik, Load(p("indexer.k_norm.weight").c_str()).data().data(),
        Load(p("indexer.k_norm.bias").c_str()).data().data(), 1e-6F);
  std::vector<float> g(kId);
  Load(p("indexer_compressor_gate.weight").c_str()).MatVec(xn, g);

  const std::uint32_t n = static_cast<std::uint32_t>(st.lat.size()) / kKv;
  st.lat.insert(st.lat.end(), lat.begin(), lat.end());
  st.ik.insert(st.ik.end(), ik.begin(), ik.end());
  st.gate.insert(st.gate.end(), g.begin(), g.end());
  if ((n + 1) % kKpool == 0) {
    const std::span<const float> ape =
        Load(p("indexer_compressor_ape.weight").c_str()).data();
    const std::uint32_t b = n + 1 - kKpool;
    std::vector<float> pnew(kId);
    for (std::uint32_t d = 0; d < kId; ++d) {
      float x[4], mx = -1e30F, sum = 0.0F;
      for (std::uint32_t s = 0; s < kKpool; ++s) {
        x[s] = st.gate[(b + s) * kId + d] + ape[d + kId * s];
        mx = std::max(mx, x[s]);
      }
      for (std::uint32_t s = 0; s < kKpool; ++s) {
        x[s] = std::exp(x[s] - mx);
        sum += x[s];
      }
      float acc = 0.0F;
      for (std::uint32_t s = 0; s < kKpool; ++s) {
        acc += (x[s] / sum) * st.ik[(b + s) * kId + d];
      }
      pnew[d] = acc;
    }
    st.pooled.insert(st.pooled.end(), pnew.begin(), pnew.end());
  }

  std::vector<float> w(kIh);
  Load(p("indexer.proj.weight").c_str()).MatVec(xn, w);
  const float wscale = 1.0F / std::sqrt(static_cast<float>(kId * kIh));
  for (auto& v : w)
    v *= wscale;

  const std::uint32_t npool =
      static_cast<std::uint32_t>(st.pooled.size()) / kId;
  std::vector<float> score(npool, 0.0F);
  for (std::uint32_t pp = 0; pp < npool; ++pp) {
    const float* kp = st.pooled.data() + static_cast<std::size_t>(pp) * kId;
    for (std::uint32_t h = 0; h < kIh; ++h) {
      const float* qh = iq.data() + static_cast<std::size_t>(h) * kId;
      double dot = 0.0;
      for (std::uint32_t d = 0; d < kId; ++d) {
        dot += static_cast<double>(qh[d]) * kp[d];
      }
      score[pp] += std::max(0.0F, static_cast<float>(dot)) * w[h];
    }
  }
  const std::uint32_t n_top = std::min(npool, kTopPool);
  std::vector<std::uint32_t> order(npool);
  std::iota(order.begin(), order.end(), 0U);
  std::stable_sort(
      order.begin(), order.end(),
      [&](std::uint32_t a, std::uint32_t b) { return score[a] > score[b]; });
  std::vector<std::uint32_t> vis;
  for (std::uint32_t i = 0; i < n_top; ++i) {
    for (std::uint32_t s = 0; s < kKpool; ++s) {
      vis.push_back(order[i] * kKpool + s);
    }
  }
  const std::uint32_t n_tail = (n + 1) % kKpool;
  for (std::uint32_t k = 0; k < n_tail; ++k) {
    vis.push_back(n - k);
  }

  std::vector<float> out(kNh * kQk, 0.0F);
  {
    const float scale = 1.0F / std::sqrt(static_cast<float>(kQk));
    const F32& wk = Load(p("attn_k_b.weight").c_str());
    const F32& wv = Load(p("attn_v_b.weight").c_str());
    std::vector<float> qabs(kKv), ctx(kKv), sc(vis.size());
    for (std::uint32_t h = 0; h < kNh; ++h) {
      wk.MatVecRows(static_cast<std::uint64_t>(h) * kKv, kKv,
                    std::span(q.data() + h * kQk, kQk), qabs);
      float mx = -1e30F;
      for (std::size_t j = 0; j < vis.size(); ++j) {
        const float* lp =
            st.lat.data() + static_cast<std::size_t>(vis[j]) * kKv;
        double dot = 0.0;
        for (std::uint32_t c = 0; c < kKv; ++c) {
          dot += static_cast<double>(qabs[c]) * lp[c];
        }
        sc[j] = static_cast<float>(dot) * scale;
        mx = std::max(mx, sc[j]);
      }
      double denom = 0.0;
      for (auto& v : sc) {
        v = std::exp(v - mx);
        denom += v;
      }
      std::fill(ctx.begin(), ctx.end(), 0.0F);
      for (std::size_t j = 0; j < vis.size(); ++j) {
        const float a = static_cast<float>(sc[j] / denom);
        const float* lp =
            st.lat.data() + static_cast<std::size_t>(vis[j]) * kKv;
        for (std::uint32_t c = 0; c < kKv; ++c) {
          ctx[c] += a * lp[c];
        }
      }
      wv.MatVecRows(
          static_cast<std::uint64_t>(h) * kQk, kQk, ctx,
          std::span(out.data() + static_cast<std::size_t>(h) * kQk, kQk));
    }
  }
  Dump(dump_dir, "dsa_lat", t, lat);
  Load(p("attn_output.weight").c_str()).MatVec(out, block_out);
}

// MoE FFN: sigmoid(router) probs, +exp_probs_b for selection only, top-8,
// renormalized (clamped denominator), x2.5, clamped swiglu experts, plus the
// shared expert. Mirrors llama.cpp build_moe_ffn.
void MoeFfn(std::uint32_t layer, std::span<const float> x, std::span<float> out,
            std::uint32_t t, const char* dump_dir) {
  const auto p = [layer](const char* s) {
    return "blk." + std::to_string(layer) + "." + s;
  };
  constexpr std::uint32_t kNe = 288, kUsed = 8, kFf = 2048, kH = 4096;
  std::vector<float> logits(kNe);
  Load(p("ffn_gate_inp.weight").c_str()).MatVec(x, logits);
  const float* eb = Load(p("exp_probs_b.bias").c_str()).data().data();
  std::vector<float> probs(kNe), sel(kNe);
  for (std::uint32_t e = 0; e < kNe; ++e) {
    probs[e] = Sigmoid(logits[e]);
    sel[e] = probs[e] + eb[e];
  }
  std::vector<std::uint32_t> idx(kNe);
  std::iota(idx.begin(), idx.end(), 0U);
  std::stable_sort(
      idx.begin(), idx.end(),
      [&](std::uint32_t a, std::uint32_t b) { return sel[a] > sel[b]; });
  float wsum = 0.0F;
  for (std::uint32_t k = 0; k < kUsed; ++k) {
    wsum += probs[idx[k]];
  }
  wsum = std::max(wsum, 6.103515625e-5F);

  const F32& wg = LoadRow(p("ffn_gate_exps.weight").c_str());
  const F32& wu = LoadRow(p("ffn_up_exps.weight").c_str());
  const F32& wd = LoadRow(p("ffn_down_exps.weight").c_str());
  std::vector<float> gf(kFf), uf(kFf), act(kFf), dpart(kH);
  std::vector<float> sel_dump(kUsed);
  for (std::uint32_t k = 0; k < kUsed; ++k) {
    const std::uint32_t e = idx[k];
    sel_dump[k] = static_cast<float>(e);
    const float wgt = probs[e] / wsum * 2.5F;
    wg.MatVecView(static_cast<std::uint64_t>(e) * kFf, kFf, x, gf);
    wu.MatVecView(static_cast<std::uint64_t>(e) * kFf, kFf, x, uf);
    for (std::uint32_t i = 0; i < kFf; ++i) {
      const float g = std::min(gf[i], 10.0F);
      const float u = std::clamp(uf[i], -10.0F, 10.0F);
      act[i] = g / (1.0F + std::exp(-g)) * u;
    }
    wd.MatVecView(static_cast<std::uint64_t>(e) * kH, kH, act, dpart);
    for (std::uint32_t i = 0; i < kH; ++i) {
      out[i] += wgt * dpart[i];
    }
  }
  Load(p("ffn_gate_shexp.weight").c_str()).MatVec(x, gf);
  Load(p("ffn_up_shexp.weight").c_str()).MatVec(x, uf);
  for (std::uint32_t i = 0; i < kFf; ++i) {
    const float g = std::min(gf[i], 10.0F);
    const float u = std::clamp(uf[i], -10.0F, 10.0F);
    act[i] = g / (1.0F + std::exp(-g)) * u;
  }
  Load(p("ffn_down_shexp.weight").c_str()).MatVec(act, dpart);
  for (std::uint32_t i = 0; i < kH; ++i) {
    out[i] += dpart[i];
  }
  Dump(dump_dir, "moe_sel", t, sel_dump);
}

}  // namespace

int main(int argc, char** argv) {
  const char* model_path = std::getenv("GUFO_GLM53_FLASH_MODEL");
  if (argc > 1 && argv[1][0] != '\0') {
    model_path = argv[1];
  }
  if (model_path == nullptr) {
    return 77;
  }
  const char* dump_dir = std::getenv("GUFO_GLM53_DUMP_DIR");
  if (dump_dir != nullptr) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dump_dir, ec);
    fs::create_directories(std::string(dump_dir) + "/pre", ec);
    fs::create_directories(std::string(dump_dir) + "/comb", ec);
    fs::create_directories(std::string(dump_dir) + "/post", ec);
  }
  std::string error;
  auto reader = GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::fprintf(stderr, "open: %s\n", error.c_str());
    return 1;
  }
  auto config = glm::ParseConfig(*reader, &error);
  if (!config) {
    std::fprintf(stderr, "config: %s\n", error.c_str());
    return 1;
  }
  for (const auto& t : reader->GetTensors()) {
    g_tensors.emplace(std::string(t.name), &t);
  }

  const std::uint32_t h = 4096, hc = 4, hd = 128, heads = 64, d_inner = 8192;
  const float hc_eps = config->hc_eps;
  const std::vector<std::int32_t> tokens = {9703, 11, 1879, 0, 101726, 98384};
  const std::uint32_t T = static_cast<std::uint32_t>(tokens.size());

  std::vector<float> x(static_cast<std::size_t>(T) * hc * h, 0.0F);
  const F32 embd = F32::RowView(g_tensors.at("token_embd.weight"));
  std::vector<float> row(h);
  for (std::uint32_t t = 0; t < T; ++t) {
    embd.DequantizeRow(static_cast<std::uint64_t>(tokens[t]), row.data());
    for (std::uint32_t s = 0; s < hc; ++s) {
      std::copy_n(row.data(), h,
                  x.data() + static_cast<std::size_t>(t) * hc * h +
                      static_cast<std::size_t>(s) * h);
    }
  }
  // Process tokens one at a time for the recurrent state.
  std::vector<std::vector<float>> tok_x(T, std::vector<float>(hc * h));
  for (std::uint32_t t = 0; t < T; ++t) {
    std::copy_n(x.data() + static_cast<std::size_t>(t) * hc * h, hc * h,
                tok_x[t].begin());
  }

  // Recurrent state per KDA layer (conv history + delta state), sequence-wide.
  struct KdaState {
    std::vector<float> conv;   // 3 * 3 * 8192 (qkv, history)
    std::vector<float> state;  // heads*hd*hd
    KdaState()
        : conv(3 * (4 - 1) * d_inner, 0.0F), state(heads * hd * hd, 0.0F) {}
  };
  std::vector<KdaState> kda_state(45);
  std::vector<DsaState> dsa_state(45);

  const std::uint32_t n_layers = std::getenv("GUFO_GLM53_LAYERS") != nullptr
                                     ? static_cast<std::uint32_t>(std::atoi(
                                           std::getenv("GUFO_GLM53_LAYERS")))
                                     : 3;

  for (std::uint32_t layer = 0; layer < n_layers; ++layer) {
    g_stage_prefix = "L" + std::to_string(layer) + "_";
    const auto p = [layer](const char* s) {
      return "blk." + std::to_string(layer) + "." + s;
    };
    const bool dsa = (layer % 4) == 3;
    for (std::uint32_t t = 0; t < T; ++t) {
      auto& cur = tok_x[t];
      // attention block
      const HcMix m_attn =
          HcPre(cur, p("hc_attn").c_str(), hc_eps, t, dump_dir);
      std::vector<float> attn_in(h), attn_in_n(h);
      for (std::uint32_t i = 0; i < h; ++i) {
        attn_in[i] = 0.0F;
        for (std::uint32_t s = 0; s < hc; ++s) {
          attn_in[i] += cur[s * h + i] * m_attn.pre[s];
        }
      }
      std::copy_n(attn_in.data(), h, attn_in_n.data());
      const float* attn_norm =
          Load(p("attn_norm.weight").c_str()).data().data();
      Rms(attn_in_n, attn_norm, 1e-5F);
      Dump(dump_dir, "attn_in", t, attn_in_n);

      std::vector<float> block_out(h);
      if (dsa) {
        DsaFwd(layer, t, attn_in_n, block_out, dsa_state[layer], dump_dir);
      } else {
        // KDA
        auto& ks = kda_state[layer];
        const float qkv_scale = 1.0F / std::sqrt(static_cast<float>(hd));
        std::vector<float> attn_out(d_inner);
        // Precompute per-token projections handled inside loop below.
        // q/k/v projections [d_inner]
        auto proj = [&](const char* w, std::vector<float>& out) {
          Load(p(w).c_str()).MatVec(attn_in_n, out);
        };
        std::vector<float> q(d_inner), k(d_inner), v(d_inner);
        proj("attn_q.weight", q);
        proj("attn_k.weight", k);
        proj("attn_v.weight", v);
        // Depthwise causal conv, SiLU, rolling history per (which, tap,
        // channel).
        std::vector<std::vector<float>*> qkv = {&q, &k, &v};
        const char* conv_names[3] = {"ssm_conv1d_q.weight",
                                     "ssm_conv1d_k.weight",
                                     "ssm_conv1d_v.weight"};
        for (std::uint32_t w = 0; w < 3; ++w) {
          const std::size_t base = w * 3 * d_inner;
          const std::vector<float>& cw = Load(p(conv_names[w]).c_str()).data();
          std::vector<float> raw = *qkv[w];
          for (std::uint32_t ch = 0; ch < d_inner; ++ch) {
            const float* tap_w = cw.data() + static_cast<std::size_t>(ch) * 4;
            float acc = tap_w[3] * raw[ch];
            for (std::uint32_t tap = 0; tap < 3; ++tap) {
              acc += tap_w[tap] * ks.conv[base + tap * d_inner + ch];
            }
            (*qkv[w])[ch] = acc / (1.0F + std::exp(-acc));
          }
          std::copy(
              ks.conv.begin() + static_cast<std::ptrdiff_t>(base) + d_inner,
              ks.conv.begin() + static_cast<std::ptrdiff_t>(base) + 3 * d_inner,
              ks.conv.begin() + static_cast<std::ptrdiff_t>(base));
          std::memcpy(ks.conv.data() + base + 2 * d_inner, raw.data(),
                      d_inner * sizeof(float));
        }
        // gate g = -5 * sigmoid(-(f_b(f_a x) + dt) * a[h])
        std::vector<float> fa(128);
        Load(p("ssm_f_a.weight").c_str()).MatVec(attn_in_n, fa);
        std::vector<float> gproj(d_inner);
        Load(p("ssm_f_b.weight").c_str()).MatVec(fa, gproj);
        const float* dt = Load(p("ssm_dt.bias").c_str()).data().data();
        const float* sa = Load(p("ssm_a").c_str()).data().data();
        std::vector<float> betah(heads);
        {
          std::vector<float> braw(heads);
          Load(p("ssm_beta.weight").c_str()).MatVec(attn_in_n, braw);
          for (std::uint32_t hh = 0; hh < heads; ++hh) {
            betah[hh] = Sigmoid(braw[hh]);
          }
        }
        const float* norm_w = Load(p("ssm_norm.weight").c_str()).data().data();
        std::vector<float> ga(128);
        Load(p("ssm_g_a.weight").c_str()).MatVec(attn_in_n, ga);
        std::vector<float> g2(d_inner);
        Load(p("ssm_g_b.weight").c_str()).MatVec(ga, g2);
        for (std::uint32_t hh = 0; hh < heads; ++hh) {
          std::span<float> qh(q.data() + hh * hd, hd);
          std::span<float> kh(k.data() + hh * hd, hd);
          L2(qh, 1e-6F);
          L2(kh, 1e-6F);
          float* S = ks.state.data() + static_cast<std::size_t>(hh) * hd * hd;
          const float a = sa[hh];
          // S[:,i] *= exp(g[i]) where g = -5*sigmoid(-(f*a)) per key dim
          std::vector<float> decay(hd);
          for (std::uint32_t i = 0; i < hd; ++i) {
            const float f = gproj[hh * hd + i] + dt[hh * hd + i];
            decay[i] = std::exp(-5.0F * (1.0F / (1.0F + std::exp(f * a))));
          }
          std::vector<float> u(hd);
          for (std::uint32_t jv = 0; jv < hd; ++jv) {
            float* row = S + static_cast<std::size_t>(jv) * hd;
            double acc = 0.0;
            for (std::uint32_t i = 0; i < hd; ++i) {
              row[i] *= decay[i];
              acc += static_cast<double>(row[i]) * kh[i];
            }
            u[jv] = static_cast<float>(acc);
          }
          float* o = attn_out.data() + hh * hd;
          for (std::uint32_t jv = 0; jv < hd; ++jv) {
            const float delta = (v[hh * hd + jv] - u[jv]) * betah[hh];
            float* row = S + static_cast<std::size_t>(jv) * hd;
            double acc = 0.0;
            for (std::uint32_t i = 0; i < hd; ++i) {
              row[i] += delta * kh[i];
              acc += static_cast<double>(row[i]) * qh[i];
            }
            o[jv] = static_cast<float>(acc) * qkv_scale;
          }
          std::span<float> os(o, hd);
          Rms(os, norm_w, 1e-5F);
          for (std::uint32_t jv = 0; jv < hd; ++jv) {
            o[jv] *= Sigmoid(g2[hh * hd + jv]);
          }
        }
        Load(p("attn_output.weight").c_str()).MatVec(attn_out, block_out);
      }
      Dump(dump_dir, "attn_block", t, block_out);
      std::vector<float> next;
      HcPost(block_out, tok_x[t], m_attn, next);
      tok_x[t].swap(next);

      // ffn block (dense)
      const HcMix m_ffn =
          HcPre(tok_x[t], p("hc_ffn").c_str(), hc_eps, t, dump_dir);
      std::vector<float> ffn_in(h);
      for (std::uint32_t i = 0; i < h; ++i) {
        ffn_in[i] = 0.0F;
        for (std::uint32_t s = 0; s < hc; ++s) {
          ffn_in[i] += tok_x[t][s * h + i] * m_ffn.pre[s];
        }
      }
      const float* ffn_norm = Load(p("ffn_norm.weight").c_str()).data().data();
      Rms(ffn_in, ffn_norm, 1e-5F);
      Dump(dump_dir, "ffn_in", t, ffn_in);
      std::vector<float> ffn_out(h, 0.0F);
      if (!dsa) {
        std::vector<float> gate(12288), up(12288), act(12288);
        Load(p("ffn_gate.weight").c_str()).MatVec(ffn_in, gate);
        Load(p("ffn_up.weight").c_str()).MatVec(ffn_in, up);
        for (std::uint32_t i = 0; i < 12288; ++i) {
          const float g = std::min(gate[i], 10.0F);
          const float u = std::clamp(up[i], -10.0F, 10.0F);
          act[i] = g / (1.0F + std::exp(-g)) * u;
        }
        Load(p("ffn_down.weight").c_str()).MatVec(act, ffn_out);
      } else {
        MoeFfn(layer, ffn_in, ffn_out, t, dump_dir);
      }
      Dump(dump_dir, "ffn_block", t, ffn_out);
      HcPost(ffn_out, tok_x[t], m_ffn, next);
      tok_x[t].swap(next);
      Dump(dump_dir, "ffn_stream", t, tok_x[t]);
    }
    std::printf("layer %u ok\n", layer);
  }

  // head: mean over streams, output_norm
  std::vector<float> head(h), xn(h);
  for (std::uint32_t t = 0; t < T; ++t) {
    for (std::uint32_t i = 0; i < h; ++i) {
      float acc = 0.0F;
      for (std::uint32_t s = 0; s < hc; ++s) {
        acc += tok_x[t][s * h + i];
      }
      head[i] = acc / static_cast<float>(hc);
    }
    std::copy_n(head.data(), h, xn.data());
    Rms(xn, Load("output_norm.weight").data().data(), 1e-5F);
    g_stage_prefix.clear();
    Dump(dump_dir, "h_final", t, xn);
    double max_abs = 0.0, sum = 0.0;
    bool finite = true;
    for (const auto v : xn) {
      if (!std::isfinite(v))
        finite = false;
      max_abs = std::max(max_abs, static_cast<double>(std::abs(v)));
      sum += v;
    }
    std::printf("h_final t=%u finite=%d max=%.6f mean=%.6f\n", t,
                finite ? 1 : 0, max_abs, sum / h);
    if (!finite || max_abs == 0.0) {
      return 1;
    }
  }
  return 0;
}
