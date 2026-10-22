#include "src/models/glm53_flash/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::glm53_flash {
namespace {

using gufo::core::GgmlType;
using gufo::core::GgufTensorInfo;

constexpr std::uint32_t kH = 4096, kHc = 4, kHd = 128, kHeads = 64;
constexpr std::uint32_t kInner = 8192, kFfn = 12288, kExpert = 2048;
constexpr std::uint32_t kExperts = 288, kUsed = 8;
constexpr std::uint32_t kKv = 512, kQk = 256, kNh = 64, kId = 128, kIh = 32;
constexpr std::uint32_t kKpool = 4, kTopPool = 2048 / kKpool;
constexpr std::uint32_t kQlora = 1536;

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

// LLM_NORM: mean-centering LayerNorm (llama.cpp ggml_norm), used by the DSA
// indexer k_norm.
void Lnorm(std::span<float> x, const float* w, const float* b, float eps) {
  double sum = 0.0;
  for (const auto v : x)
    sum += static_cast<double>(v);
  const double mu = sum / static_cast<double>(x.size());
  double s2 = 0.0;
  for (const auto v : x) {
    const double d = static_cast<double>(v) - mu;
    s2 += d * d;
  }
  const float inv = static_cast<float>(
      1.0 / std::sqrt(s2 / static_cast<double>(x.size()) + eps));
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] =
        (static_cast<double>(x[i]) - mu) * inv * (w != nullptr ? w[i] : 1.0F) +
        (b != nullptr ? b[i] : 0.0F);
  }
}

void Swiglu(std::span<const float> gate, std::span<const float> up,
            std::span<float> act) {
  for (std::size_t i = 0; i < act.size(); ++i) {
    const float g = std::min(gate[i], 10.0F);
    const float u = std::clamp(up[i], -10.0F, 10.0F);
    act[i] = g * Sigmoid(g) * u;
  }
}

std::string P(std::uint32_t layer, const char* s) {
  return "blk." + std::to_string(layer) + "." + s;
}

}  // namespace

// F32 tensor materialized on demand (or row-view only for huge tensors).
ReferenceModel::Loaded::Loaded(const GgufTensorInfo* t) {
  Setup(t);
  data_.resize(static_cast<std::size_t>(rows_) * cols_);
  for (std::uint64_t r = 0; r < rows_; ++r) {
    DequantizeRow(r, data_.data() + r * cols_);
  }
}

ReferenceModel::Loaded ReferenceModel::Loaded::RowView(
    const GgufTensorInfo* t) {
  Loaded f;
  f.Setup(t);
  return f;
}

void ReferenceModel::Loaded::Setup(const GgufTensorInfo* t) {
  t_ = t;
  rows_ = 1;
  for (std::size_t d = 1; d < t->dimensions.size(); ++d) {
    rows_ *= t->dimensions[d];
  }
  cols_ = t->dimensions[0];
}

void ReferenceModel::Loaded::DequantizeRow(std::uint64_t row,
                                           float* out) const {
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
      std::abort();
  }
}

void ReferenceModel::Loaded::MatVec(std::span<const float> x,
                                    std::span<float> y) const {
  for (std::uint64_t r = 0; r < rows_; ++r) {
    const float* w = data_.data() + r * cols_;
    double acc = 0.0;
    for (std::uint64_t i = 0; i < cols_; ++i) {
      acc += static_cast<double>(w[i]) * x[i];
    }
    y[r] = static_cast<float>(acc);
  }
}

void ReferenceModel::Loaded::MatVecRows(std::uint64_t row0, std::uint64_t nrows,
                                        std::span<const float> x,
                                        std::span<float> y) const {
  for (std::uint64_t r = 0; r < nrows; ++r) {
    const float* w = data_.data() + (row0 + r) * cols_;
    double acc = 0.0;
    for (std::uint64_t i = 0; i < cols_; ++i) {
      acc += static_cast<double>(w[i]) * x[i];
    }
    y[r] = static_cast<float>(acc);
  }
}

void ReferenceModel::Loaded::MatVecView(std::uint64_t row0, std::uint64_t nrows,
                                        std::span<const float> x,
                                        std::span<float> y) const {
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

std::unique_ptr<ReferenceModel> ReferenceModel::Open(const std::string& path,
                                                     std::string* error) {
  auto model = std::unique_ptr<ReferenceModel>(new ReferenceModel());
  model->reader_ = gufo::core::GgufReader::OpenFile(path, error);
  if (!model->reader_) {
    return nullptr;
  }
  auto config = ParseConfig(*model->reader_, error);
  if (!config) {
    return nullptr;
  }
  model->config_ = *config;
  for (const auto& t : model->reader_->GetTensors()) {
    model->tensors_.emplace(std::string(t.name), &t);
  }
  return model;
}

const ReferenceModel::Loaded& ReferenceModel::Load(
    const std::string& name) const {
  auto it = cache_.find(name);
  if (it != cache_.end()) {
    return it->second;
  }
  const auto found = tensors_.find(name);
  if (found == tensors_.end()) {
    std::abort();
  }
  return cache_.emplace(name, Loaded(found->second)).first->second;
}

const ReferenceModel::Loaded& ReferenceModel::LoadRow(
    const std::string& name) const {
  auto it = cache_.find(name);
  if (it != cache_.end()) {
    return it->second;
  }
  const auto found = tensors_.find(name);
  if (found == tensors_.end()) {
    std::abort();
  }
  return cache_.emplace(name, Loaded::RowView(found->second)).first->second;
}

std::uint32_t ReferenceModel::n_layers() const {
  return config_.num_layers - config_.nextn_layers;
}

std::unique_ptr<ReferenceModel::Session> ReferenceModel::CreateSession() const {
  auto s = std::make_unique<Session>();
  s->x.assign(kHc * kH, 0.0F);
  s->layers.resize(n_layers());
  for (auto& l : s->layers) {
    l.conv.assign(3 * 3 * kInner, 0.0F);
    l.state.assign(kHeads * kHd * kHd, 0.0F);
  }
  return s;
}

void ReferenceModel::HcMix(std::span<const float> x, const std::string& prefix,
                           float eps, std::vector<float>& pre,
                           std::vector<float>& post,
                           std::vector<float>& comb) const {
  std::vector<float> xn(x.begin(), x.end());
  Rms(xn, nullptr, 1e-5F);
  std::vector<float> mixes(24, 0.0F);
  Load(prefix + "_fn.weight").MatVec(xn, mixes);
  const float* scale = Load(prefix + "_scale.weight").w();
  const float* base = Load(prefix + "_base.weight").w();
  pre.assign(kHc, 0.0F);
  post.assign(kHc, 0.0F);
  comb.assign(kHc * kHc, 0.0F);
  for (std::uint32_t s = 0; s < kHc; ++s) {
    pre[s] = Sigmoid(mixes[s] * scale[0] + base[s]) + eps;
    post[s] = Sigmoid(mixes[kHc + s] * scale[1] + base[kHc + s]) * 2.0F;
  }
  for (std::uint32_t i = 0; i < kHc * kHc; ++i) {
    comb[i] = mixes[2 * kHc + i] * scale[2] + base[2 * kHc + i];
  }
  // comb[dst + hc*src]; softmax over dst, then sinkhorn.
  for (std::uint32_t src = 0; src < kHc; ++src) {
    double mx = -1e30, sum = 0.0;
    for (std::uint32_t d = 0; d < kHc; ++d) {
      mx = std::max(mx, static_cast<double>(comb[d + kHc * src]));
    }
    for (std::uint32_t d = 0; d < kHc; ++d) {
      comb[d + kHc * src] =
          static_cast<float>(std::exp(comb[d + kHc * src] - mx));
      sum += comb[d + kHc * src];
    }
    for (std::uint32_t d = 0; d < kHc; ++d) {
      comb[d + kHc * src] /= static_cast<float>(sum + eps);
    }
  }
  const auto norm = [&](bool over_src) {
    for (std::uint32_t a = 0; a < kHc; ++a) {
      double sum = 0.0;
      for (std::uint32_t b = 0; b < kHc; ++b) {
        sum += over_src ? comb[a + kHc * b] : comb[b + kHc * a];
      }
      const float denom = static_cast<float>(sum + eps);
      for (std::uint32_t b = 0; b < kHc; ++b) {
        if (over_src) {
          comb[a + kHc * b] /= denom;
        } else {
          comb[b + kHc * a] /= denom;
        }
      }
    }
  };
  norm(true);
  for (int i = 0; i < 19; ++i) {
    norm(false);
    norm(true);
  }
}

void ReferenceModel::AttnBlock(std::uint32_t layer, Session& session, bool dsa,
                               std::span<const float> xn,
                               std::span<float> block) const {
  auto& st = session.layers[layer];
  if (!dsa) {
    // KDA linear attention.
    const float qkv_scale = 1.0F / std::sqrt(static_cast<float>(kHd));
    std::vector<float> attn_out(kInner);
    std::vector<float> q(kInner), k(kInner), v(kInner);
    Load(P(layer, "attn_q.weight")).MatVec(xn, q);
    Load(P(layer, "attn_k.weight")).MatVec(xn, k);
    Load(P(layer, "attn_v.weight")).MatVec(xn, v);
    std::vector<std::vector<float>*> qkv = {&q, &k, &v};
    const char* conv_names[3] = {"ssm_conv1d_q.weight", "ssm_conv1d_k.weight",
                                 "ssm_conv1d_v.weight"};
    for (std::uint32_t w = 0; w < 3; ++w) {
      const std::size_t base = w * 3 * kInner;
      const std::vector<float>& cw = Load(P(layer, conv_names[w])).data();
      std::vector<float> raw = *qkv[w];
      for (std::uint32_t ch = 0; ch < kInner; ++ch) {
        const float* tap_w = cw.data() + static_cast<std::size_t>(ch) * 4;
        float acc = tap_w[3] * raw[ch];
        for (std::uint32_t tap = 0; tap < 3; ++tap) {
          acc += tap_w[tap] * st.conv[base + tap * kInner + ch];
        }
        (*qkv[w])[ch] = acc * Sigmoid(acc);
      }
      std::copy(
          st.conv.begin() + static_cast<std::ptrdiff_t>(base) + kInner,
          st.conv.begin() + static_cast<std::ptrdiff_t>(base) + 3 * kInner,
          st.conv.begin() + static_cast<std::ptrdiff_t>(base));
      std::memcpy(st.conv.data() + base + 2 * kInner, raw.data(),
                  kInner * sizeof(float));
    }
    std::vector<float> fa(kHeads);
    Load(P(layer, "ssm_f_a.weight")).MatVec(xn, fa);
    std::vector<float> gproj(kInner);
    Load(P(layer, "ssm_f_b.weight")).MatVec(fa, gproj);
    const float* dt = Load(P(layer, "ssm_dt.bias")).w();
    const float* sa = Load(P(layer, "ssm_a")).w();
    std::vector<float> betah(kHeads);
    {
      std::vector<float> braw(kHeads);
      Load(P(layer, "ssm_beta.weight")).MatVec(xn, braw);
      for (std::uint32_t hh = 0; hh < kHeads; ++hh) {
        betah[hh] = Sigmoid(braw[hh]);
      }
    }
    const float* norm_w = Load(P(layer, "ssm_norm.weight")).w();
    std::vector<float> ga(kHeads);
    Load(P(layer, "ssm_g_a.weight")).MatVec(xn, ga);
    std::vector<float> g2(kInner);
    Load(P(layer, "ssm_g_b.weight")).MatVec(ga, g2);
    for (std::uint32_t hh = 0; hh < kHeads; ++hh) {
      std::span<float> qh(q.data() + hh * kHd, kHd);
      std::span<float> kh(k.data() + hh * kHd, kHd);
      L2(qh, 1e-6F);
      L2(kh, 1e-6F);
      float* s_mat = st.state.data() + static_cast<std::size_t>(hh) * kHd * kHd;
      const float a = sa[hh];
      std::vector<float> decay(kHd);
      for (std::uint32_t i = 0; i < kHd; ++i) {
        const float f = gproj[hh * kHd + i] + dt[hh * kHd + i];
        decay[i] = std::exp(-5.0F * (1.0F / (1.0F + std::exp(f * a))));
      }
      std::vector<float> u(kHd);
      for (std::uint32_t jv = 0; jv < kHd; ++jv) {
        float* srow = s_mat + static_cast<std::size_t>(jv) * kHd;
        double acc = 0.0;
        for (std::uint32_t i = 0; i < kHd; ++i) {
          srow[i] *= decay[i];
          acc += static_cast<double>(srow[i]) * kh[i];
        }
        u[jv] = static_cast<float>(acc);
      }
      float* o = attn_out.data() + hh * kHd;
      for (std::uint32_t jv = 0; jv < kHd; ++jv) {
        const float delta = (v[hh * kHd + jv] - u[jv]) * betah[hh];
        float* srow = s_mat + static_cast<std::size_t>(jv) * kHd;
        double acc = 0.0;
        for (std::uint32_t i = 0; i < kHd; ++i) {
          srow[i] += delta * kh[i];
          acc += static_cast<double>(srow[i]) * qh[i];
        }
        o[jv] = static_cast<float>(acc) * qkv_scale;
      }
      std::span<float> os(o, kHd);
      Rms(os, norm_w, 1e-5F);
      for (std::uint32_t jv = 0; jv < kHd; ++jv) {
        o[jv] *= Sigmoid(g2[hh * kHd + jv]);
      }
    }
    Load(P(layer, "attn_output.weight")).MatVec(attn_out, block);
    return;
  }

  // Absorbed MLA + kpool indexer.
  std::vector<float> qr(kQlora), q(kNh * kQk), lat(kKv);
  Load(P(layer, "attn_q_a.weight")).MatVec(xn, qr);
  Rms(qr, Load(P(layer, "attn_q_a_norm.weight")).w(), 1e-5F);
  Load(P(layer, "attn_q_b.weight")).MatVec(qr, q);
  Load(P(layer, "attn_kv_a_mqa.weight")).MatVec(xn, lat);
  Rms(lat, Load(P(layer, "attn_kv_a_norm.weight")).w(), 1e-5F);

  std::vector<float> iq(kIh * kId);
  Load(P(layer, "indexer.attn_q_b.weight")).MatVec(qr, iq);
  std::vector<float> ik(kId);
  Load(P(layer, "indexer.attn_k.weight")).MatVec(xn, ik);
  Lnorm(ik, Load(P(layer, "indexer.k_norm.weight")).w(),
        Load(P(layer, "indexer.k_norm.bias")).w(), 1e-6F);
  std::vector<float> g(kId);
  Load(P(layer, "indexer_compressor_gate.weight")).MatVec(xn, g);

  const std::uint32_t n = static_cast<std::uint32_t>(st.lat.size()) / kKv;
  st.lat.insert(st.lat.end(), lat.begin(), lat.end());
  st.ik.insert(st.ik.end(), ik.begin(), ik.end());
  st.gate.insert(st.gate.end(), g.begin(), g.end());
  if ((n + 1) % kKpool == 0) {
    const std::span<const float> ape =
        Load(P(layer, "indexer_compressor_ape.weight")).data();
    const std::uint32_t b = n + 1 - kKpool;
    std::vector<float> pnew(kId);
    for (std::uint32_t d = 0; d < kId; ++d) {
      float x4[4], mx = -1e30F, sum = 0.0F;
      for (std::uint32_t s = 0; s < kKpool; ++s) {
        x4[s] = st.gate[(b + s) * kId + d] + ape[d + kId * s];
        mx = std::max(mx, x4[s]);
      }
      for (std::uint32_t s = 0; s < kKpool; ++s) {
        x4[s] = std::exp(x4[s] - mx);
        sum += x4[s];
      }
      float acc = 0.0F;
      for (std::uint32_t s = 0; s < kKpool; ++s) {
        acc += (x4[s] / sum) * st.ik[(b + s) * kId + d];
      }
      pnew[d] = acc;
    }
    st.pooled.insert(st.pooled.end(), pnew.begin(), pnew.end());
  }

  std::vector<float> w(kIh);
  Load(P(layer, "indexer.proj.weight")).MatVec(xn, w);
  const float wscale = 1.0F / std::sqrt(static_cast<float>(kId * kIh));
  for (auto& v : w)
    v *= wscale;

  const std::uint32_t npool =
      static_cast<std::uint32_t>(st.pooled.size()) / kId;
  std::vector<float> score(npool, 0.0F);
  for (std::uint32_t pp = 0; pp < npool; ++pp) {
    const float* kp = st.pooled.data() + static_cast<std::size_t>(pp) * kId;
    double acc = 0.0;
    for (std::uint32_t h = 0; h < kIh; ++h) {
      const float* qh = iq.data() + static_cast<std::size_t>(h) * kId;
      double dot = 0.0;
      for (std::uint32_t d = 0; d < kId; ++d) {
        dot += static_cast<double>(qh[d]) * kp[d];
      }
      acc +=
          std::max(0.0F, static_cast<float>(dot)) * static_cast<double>(w[h]);
    }
    score[pp] = static_cast<float>(acc);
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
  const float scale = 1.0F / std::sqrt(static_cast<float>(kQk));
  const Loaded& wk = Load(P(layer, "attn_k_b.weight"));
  const Loaded& wv = Load(P(layer, "attn_v_b.weight"));
  std::vector<float> qabs(kKv), ctx(kKv), sc(vis.size());
  for (std::uint32_t h = 0; h < kNh; ++h) {
    wk.MatVecRows(static_cast<std::uint64_t>(h) * kKv, kKv,
                  std::span(q.data() + h * kQk, kQk), qabs);
    float mx = -1e30F;
    for (std::size_t j = 0; j < vis.size(); ++j) {
      const float* lp = st.lat.data() + static_cast<std::size_t>(vis[j]) * kKv;
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
      const float* lp = st.lat.data() + static_cast<std::size_t>(vis[j]) * kKv;
      for (std::uint32_t c = 0; c < kKv; ++c) {
        ctx[c] += a * lp[c];
      }
    }
    wv.MatVecRows(
        static_cast<std::uint64_t>(h) * kQk, kQk, ctx,
        std::span(out.data() + static_cast<std::size_t>(h) * kQk, kQk));
  }
  Load(P(layer, "attn_output.weight")).MatVec(out, block);
}

void ReferenceModel::FfnBlock(std::uint32_t layer, std::span<const float> fin,
                              std::span<float> ffn_out) const {
  if (layer < 3) {
    std::vector<float> gate(kFfn), up(kFfn), act(kFfn);
    Load(P(layer, "ffn_gate.weight")).MatVec(fin, gate);
    Load(P(layer, "ffn_up.weight")).MatVec(fin, up);
    Swiglu(gate, up, act);
    Load(P(layer, "ffn_down.weight")).MatVec(act, ffn_out);
    return;
  }
  std::fill(ffn_out.begin(), ffn_out.end(), 0.0F);
  std::vector<float> logits(kExperts);
  Load(P(layer, "ffn_gate_inp.weight")).MatVec(fin, logits);
  const float* eb = Load(P(layer, "exp_probs_b.bias")).w();
  std::vector<float> probs(kExperts), sel(kExperts);
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    probs[e] = Sigmoid(logits[e]);
    sel[e] = probs[e] + eb[e];
  }
  std::vector<std::uint32_t> idx(kExperts);
  std::iota(idx.begin(), idx.end(), 0U);
  std::stable_sort(
      idx.begin(), idx.end(),
      [&](std::uint32_t a, std::uint32_t b) { return sel[a] > sel[b]; });
  float wsum = 0.0F;
  for (std::uint32_t i = 0; i < kUsed; ++i) {
    wsum += probs[idx[i]];
  }
  wsum = std::max(wsum, 6.103515625e-5F);
  const Loaded& wg = LoadRow(P(layer, "ffn_gate_exps.weight"));
  const Loaded& wu = LoadRow(P(layer, "ffn_up_exps.weight"));
  const Loaded& wd = LoadRow(P(layer, "ffn_down_exps.weight"));
  std::vector<float> gf(kExpert), uf(kExpert), act(kExpert), dpart(kH);
  for (std::uint32_t i = 0; i < kUsed; ++i) {
    const std::uint32_t e = idx[i];
    const float wgt = probs[e] / wsum * 2.5F;
    wg.MatVecView(static_cast<std::uint64_t>(e) * kExpert, kExpert, fin, gf);
    wu.MatVecView(static_cast<std::uint64_t>(e) * kExpert, kExpert, fin, uf);
    Swiglu(gf, uf, act);
    wd.MatVecView(static_cast<std::uint64_t>(e) * kH, kH, act, dpart);
    for (std::uint32_t c = 0; c < kH; ++c) {
      ffn_out[c] += wgt * dpart[c];
    }
  }
  std::vector<float> sg(kExpert), su(kExpert);
  Load(P(layer, "ffn_gate_shexp.weight")).MatVec(fin, sg);
  Load(P(layer, "ffn_up_shexp.weight")).MatVec(fin, su);
  Swiglu(sg, su, act);
  Load(P(layer, "ffn_down_shexp.weight")).MatVec(act, dpart);
  for (std::uint32_t c = 0; c < kH; ++c) {
    ffn_out[c] += dpart[c];
  }
}

const std::vector<float>& ReferenceModel::Evaluate(Session& session,
                                                   std::int32_t token) const {
  const GgufTensorInfo& embd = *tensors_.at("token_embd.weight");
  const Loaded row_view = Loaded::RowView(&embd);
  std::vector<float> row(kH);
  row_view.DequantizeRow(static_cast<std::uint64_t>(token), row.data());
  for (std::uint32_t s = 0; s < kHc; ++s) {
    std::copy_n(row.data(), kH, session.x.data() + s * kH);
  }

  const float eps = config_.hc_eps;
  for (std::uint32_t layer = 0; layer < n_layers(); ++layer) {
    const bool dsa = (layer % 4) == 3;

    std::vector<float> pre, post, comb;
    HcMix(session.x, P(layer, "hc_attn"), eps, pre, post, comb);
    std::vector<float> attn_in(kH);
    for (std::uint32_t i = 0; i < kH; ++i) {
      attn_in[i] = 0.0F;
      for (std::uint32_t s = 0; s < kHc; ++s) {
        attn_in[i] += session.x[s * kH + i] * pre[s];
      }
    }
    Rms(attn_in, Load(P(layer, "attn_norm.weight")).w(), 1e-5F);
    std::vector<float> block(kH);
    AttnBlock(layer, session, dsa, attn_in, block);
    std::vector<float> next;
    {
      next.assign(session.x.size(), 0.0F);
      for (std::uint32_t d = 0; d < kHc; ++d) {
        for (std::uint32_t i = 0; i < kH; ++i) {
          float acc = block[i] * post[d];
          for (std::uint32_t s = 0; s < kHc; ++s) {
            acc += session.x[s * kH + i] * comb[d + kHc * s];
          }
          next[d * kH + i] = acc;
        }
      }
    }
    session.x.swap(next);

    HcMix(session.x, P(layer, "hc_ffn"), eps, pre, post, comb);
    std::vector<float> ffn_in(kH);
    for (std::uint32_t i = 0; i < kH; ++i) {
      ffn_in[i] = 0.0F;
      for (std::uint32_t s = 0; s < kHc; ++s) {
        ffn_in[i] += session.x[s * kH + i] * pre[s];
      }
    }
    Rms(ffn_in, Load(P(layer, "ffn_norm.weight")).w(), 1e-5F);
    std::vector<float> ffn_out(kH, 0.0F);
    FfnBlock(layer, ffn_in, ffn_out);
    {
      next.assign(session.x.size(), 0.0F);
      for (std::uint32_t d = 0; d < kHc; ++d) {
        for (std::uint32_t i = 0; i < kH; ++i) {
          float acc = ffn_out[i] * post[d];
          for (std::uint32_t s = 0; s < kHc; ++s) {
            acc += session.x[s * kH + i] * comb[d + kHc * s];
          }
          next[d * kH + i] = acc;
        }
      }
    }
    session.x.swap(next);
    ++session.position;
  }

  std::vector<float> head(kH);
  for (std::uint32_t i = 0; i < kH; ++i) {
    float acc = 0.0F;
    for (std::uint32_t s = 0; s < kHc; ++s) {
      acc += session.x[s * kH + i];
    }
    head[i] = acc / static_cast<float>(kHc);
  }
  Rms(head, Load("output_norm.weight").w(), 1e-5F);
  const Loaded& head_w = Load("output.weight");
  session.logits.resize(static_cast<std::size_t>(head_w.rows()));
  head_w.MatVec(head, session.logits);
  return session.logits;
}

}  // namespace gufo::models::glm53_flash
