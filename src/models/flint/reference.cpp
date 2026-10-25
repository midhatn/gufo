#include "src/models/flint/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::flint {
namespace {

using gufo::core::GgmlType;
using gufo::core::GgufTensorInfo;

float Sigmoid(float x) {
  return 1.0F / (1.0F + std::exp(-x));
}

void RmsInplace(std::span<float> x, const float* w, float eps) {
  double sum = 0.0;
  for (const auto v : x)
    sum += static_cast<double>(v) * v;
  const float inv = static_cast<float>(
      1.0 / std::sqrt(sum / static_cast<double>(x.size()) + eps));
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = x[i] * inv * (w != nullptr ? w[i] : 1.0F);
  }
}

float Bf16ToFloat(std::uint16_t h) {
  const std::uint32_t u32 = static_cast<std::uint32_t>(h) << 16U;
  float f = 0.0F;
  std::memcpy(&f, &u32, sizeof(float));
  return f;
}

std::string P(std::uint32_t layer, const char* s) {
  return "blk." + std::to_string(layer) + "." + s;
}

}  // namespace

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
    case GgmlType::kBF16: {
      const auto* p = reinterpret_cast<const std::uint16_t*>(src);
      for (std::uint64_t i = 0; i < cols_; ++i) {
        out[i] = Bf16ToFloat(p[i]);
      }
      break;
    }
    case GgmlType::kQ4_K:
      gufo::quant::DequantizeQ4_K(src, out, cols_);
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
    case GgmlType::kIQ4_XS:
      gufo::quant::DequantizeIQ4_XS(src, out, cols_);
      break;
    case GgmlType::kIQ3_S:
      gufo::quant::DequantizeIQ3_S(src, out, cols_);
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

std::unique_ptr<ReferenceModel::Session> ReferenceModel::CreateSession() const {
  auto s = std::make_unique<Session>();
  s->x.assign(config_.hidden_size, 0.0F);
  s->logits.assign(config_.vocab_size, 0.0F);
  s->layers.resize(config_.num_layers);
  const std::size_t kv_stride =
      static_cast<std::size_t>(config_.num_kv) * config_.head_dim;
  for (auto& l : s->layers) {
    l.k.reserve(kv_stride * 64);
    l.v.reserve(kv_stride * 64);
  }
  return s;
}

void ReferenceModel::Rms(std::span<float> x, const float* w) const {
  RmsInplace(x, w, config_.rms_eps);
}

void ReferenceModel::RopeNeoX(std::span<float> q_or_k,
                              std::uint32_t pos) const {
  // NEOX: rotate pair (i, i + head_dim / 2) with inv_freq for pair i.
  const float base = config_.rope_base;
  const std::uint32_t n_heads =
      static_cast<std::uint32_t>(q_or_k.size() / config_.head_dim);
  for (std::uint32_t head = 0; head < n_heads; ++head) {
    float* base_ptr = q_or_k.data() + head * config_.head_dim;
    const std::uint32_t half = config_.head_dim / 2;
    for (std::uint32_t i = 0; i < half; ++i) {
      const float inv_freq =
          1.0F / std::pow(base, static_cast<float>(2 * i) /
                                    static_cast<float>(config_.head_dim));
      const float theta = static_cast<float>(pos) * inv_freq;
      const float c = std::cos(theta);
      const float s = std::sin(theta);
      const float x0 = base_ptr[i];
      const float x1 = base_ptr[i + half];
      base_ptr[i] = x0 * c - x1 * s;
      base_ptr[i + half] = x0 * s + x1 * c;
    }
  }
}

void ReferenceModel::Attention(std::uint32_t layer, Session& session,
                               std::span<const float> xn,
                               std::span<float> out) const {
  auto& st = session.layers[layer];
  const auto& cfg = config_;
  const std::uint32_t hd = cfg.head_dim;
  const std::uint32_t n_q = cfg.num_heads;
  const std::uint32_t n_kv = cfg.num_kv;
  const std::uint32_t pos = session.position;
  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));

  std::vector<float> q(n_q * hd), k(n_kv * hd), v(n_kv * hd);
  Load(P(layer, "attn_q.weight")).MatVec(xn, q);
  Load(P(layer, "attn_k.weight")).MatVec(xn, k);
  Load(P(layer, "attn_v.weight")).MatVec(xn, v);

  // Per-head QK RMSNorm.
  std::span<float> qn(q);
  std::span<float> kn(k);
  const float* qn_w = Load(P(layer, "attn_q_norm.weight")).w();
  const float* kn_w = Load(P(layer, "attn_k_norm.weight")).w();
  for (std::uint32_t h = 0; h < n_q; ++h) {
    Rms(qn.subspan(h * hd, hd), qn_w);
  }
  for (std::uint32_t h = 0; h < n_kv; ++h) {
    Rms(kn.subspan(h * hd, hd), kn_w);
  }

  if (cfg.HasRope(layer)) {
    RopeNeoX(q, pos);
    RopeNeoX(k, pos);
  }

  // Append KV cache (full history; SWA applies as a mask).
  const std::size_t kv_stride = static_cast<std::size_t>(n_kv) * hd;
  st.k.insert(st.k.end(), k.begin(), k.end());
  st.v.insert(st.v.end(), v.begin(), v.end());
  const std::uint32_t n_past = st.n_tokens;
  st.n_tokens = n_past + 1;

  const float* sink = Load(P(layer, "attn_sinks.weight")).w();
  const std::uint32_t window = cfg.sliding_window;
  const bool use_swa = cfg.IsSwa(layer);
  std::vector<float> scores(st.n_tokens + 1);  // +1 sink column
  std::vector<float> att(st.n_tokens);

  std::fill(out.begin(), out.end(), 0.0F);
  std::vector<float> acc(hd, 0.0F);
  for (std::uint32_t h = 0; h < n_q; ++h) {
    const float* qh = q.data() + h * hd;
    const std::uint32_t kv_head = h * n_kv / n_q;
    double max_score = -1.0e30;
    for (std::uint32_t t = 0; t < st.n_tokens; ++t) {
      if (use_swa && pos >= window && t + window <= pos) {
        scores[t] = -1.0e30F;
        continue;
      }
      const float* kh =
          st.k.data() + static_cast<std::size_t>(t) * kv_stride + kv_head * hd;
      double dot = 0.0;
      for (std::uint32_t i = 0; i < hd; ++i) {
        dot += static_cast<double>(qh[i]) * kh[i];
      }
      scores[t] = static_cast<float>(dot) * scale;
      max_score = std::max(max_score, static_cast<double>(scores[t]));
    }
    // Sink participates in softmax as an extra logit (llama
    // soft_max_add_sinks).
    scores[st.n_tokens] = sink[h];
    max_score = std::max(max_score, static_cast<double>(scores[st.n_tokens]));
    double denom = 0.0;
    for (std::uint32_t t = 0; t < st.n_tokens; ++t) {
      att[t] = std::exp(scores[t] - static_cast<float>(max_score));
      denom += att[t];
    }
    const float sink_w =
        std::exp(scores[st.n_tokens] - static_cast<float>(max_score));
    denom += sink_w;
    const float inv = 1.0F / static_cast<float>(denom);
    std::fill(acc.begin(), acc.end(), 0.0F);
    for (std::uint32_t t = 0; t < st.n_tokens; ++t) {
      const float w = att[t] * inv;
      const float* vh =
          st.v.data() + static_cast<std::size_t>(t) * kv_stride + kv_head * hd;
      for (std::uint32_t i = 0; i < hd; ++i) {
        acc[i] += w * vh[i];
      }
    }
    // Sink has no value contribution; mass only reduces other weights.
    for (std::uint32_t i = 0; i < hd; ++i) {
      out[h * hd + i] = acc[i];
    }
  }
  // Project [n_q * head_dim] -> [hidden]. Separate buffer: MatVec must not
  // alias input with output.
  std::vector<float> proj(out.size());
  Load(P(layer, "attn_output.weight")).MatVec(out, proj);
  std::copy(proj.begin(),
            proj.begin() + std::min<std::size_t>(proj.size(), out.size()),
            out.begin());
}

void ReferenceModel::DenseFfn(std::uint32_t layer, std::span<const float> in,
                              std::span<float> out) const {
  const std::uint32_t ffn = config_.dense_ff;
  std::vector<float> gate(ffn), up(ffn);
  Load(P(layer, "ffn_gate.weight")).MatVec(in, gate);
  Load(P(layer, "ffn_up.weight")).MatVec(in, up);
  for (std::uint32_t i = 0; i < ffn; ++i) {
    gate[i] = gate[i] * Sigmoid(gate[i]) * up[i];
  }
  Load(P(layer, "ffn_down.weight")).MatVec(gate, out);
}

void ReferenceModel::MoEFfn(std::uint32_t layer, Session& session,
                            std::span<const float> in,
                            std::span<float> out) const {
  const auto& cfg = config_;
  const std::uint32_t n_exp = cfg.expert_count;
  const std::uint32_t topk = cfg.expert_used;
  const std::uint32_t ffn_e = cfg.expert_ff;
  const std::uint32_t ffn_s = cfg.shared_ff;
  const std::uint32_t hidden = cfg.hidden_size;

  // Router: gate_inp [hidden, n_exp] -> logits [n_exp].
  std::vector<float> logits(n_exp);
  Load(P(layer, "ffn_gate_inp.weight")).MatVec(in, logits);
  std::vector<float> probs(n_exp);
  for (std::uint32_t e = 0; e < n_exp; ++e) {
    probs[e] = Sigmoid(logits[e]);
  }
  std::vector<std::uint32_t> idx(n_exp);
  std::iota(idx.begin(), idx.end(), 0U);
  std::partial_sort(
      idx.begin(), idx.begin() + topk, idx.end(),
      [&](std::uint32_t a, std::uint32_t b) { return probs[a] > probs[b]; });
  std::vector<float> weights(topk);
  float wsum = 0.0F;
  for (std::uint32_t i = 0; i < topk; ++i) {
    weights[i] = probs[idx[i]];
    wsum += weights[i];
  }
  if (cfg.expert_weights_norm) {
    const float denom = std::max(wsum, 6.103515625e-5F);
    for (auto& w : weights) {
      w /= denom;
    }
  }
  if (cfg.expert_weights_scale != 1.0F) {
    for (auto& w : weights) {
      w *= cfg.expert_weights_scale;
    }
  }

  // Routed experts (row-view dequant per matvec; 256x2048x512 is too large).
  std::fill(out.begin(), out.end(), 0.0F);
  const auto& gate_exps = LoadRow(P(layer, "ffn_gate_exps.weight"));
  const auto& up_exps = LoadRow(P(layer, "ffn_up_exps.weight"));
  const auto& down_exps = LoadRow(P(layer, "ffn_down_exps.weight"));
  std::vector<float> act(ffn_e), h(ffn_e), expert_out(hidden);
  for (std::uint32_t i = 0; i < topk; ++i) {
    const std::uint64_t e = idx[i];
    // Expert blocks are contiguous in dim2 (ne2=n_expert).
    const std::uint64_t row0 = e * ffn_e;
    gate_exps.MatVecView(row0, ffn_e, in, act);
    up_exps.MatVecView(row0, ffn_e, in, h);
    for (std::uint32_t j = 0; j < ffn_e; ++j) {
      act[j] = act[j] * Sigmoid(act[j]) * h[j];
    }
    // down_exps layout: [ffn_e, hidden, n_expert] -> expert e rows e*hidden.
    down_exps.MatVecView(e * hidden, hidden, act, expert_out);
    const float w = weights[i];
    for (std::uint32_t j = 0; j < hidden; ++j) {
      out[j] += w * expert_out[j];
    }
  }

  // Shared expert always on.
  std::vector<float> sgate(ffn_s), sup(ffn_s);
  Load(P(layer, "ffn_gate_shexp.weight")).MatVec(in, sgate);
  Load(P(layer, "ffn_up_shexp.weight")).MatVec(in, sup);
  for (std::uint32_t j = 0; j < ffn_s; ++j) {
    sgate[j] = sgate[j] * Sigmoid(sgate[j]) * sup[j];
  }
  std::vector<float> shared_out(hidden);
  Load(P(layer, "ffn_down_shexp.weight")).MatVec(sgate, shared_out);
  for (std::uint32_t j = 0; j < hidden; ++j) {
    out[j] += shared_out[j];
  }
}

const std::vector<float>& ReferenceModel::Evaluate(Session& session,
                                                   std::int32_t token) const {
  const auto& cfg = config_;
  const std::uint32_t hidden = cfg.hidden_size;

  // Embedding: tok_embd [hidden, vocab] -> row `token`.
  {
    const auto& emb = LoadRow("token_embd.weight");
    emb.DequantizeRow(static_cast<std::uint64_t>(token), session.x.data());
  }

  std::vector<float> attn_raw(static_cast<std::size_t>(cfg.num_heads) *
                              cfg.head_dim);
  std::vector<float> ffn_in(hidden);
  std::vector<float> ffn_out(hidden);
  std::vector<float> normed(hidden);

  for (std::uint32_t il = 0; il < cfg.num_layers; ++il) {
    std::span<const float> inp(session.x);
    (void)inp;
    std::memcpy(normed.data(), session.x.data(), hidden * sizeof(float));
    Rms(normed, Load(P(il, "attn_norm.weight")).w());
    Attention(il, session, normed, attn_raw);
    for (std::uint32_t i = 0; i < hidden; ++i) {
      ffn_in[i] = session.x[i] + attn_raw[i];
    }
    std::memcpy(normed.data(), ffn_in.data(), hidden * sizeof(float));
    Rms(normed, Load(P(il, "ffn_norm.weight")).w());
    if (il < cfg.leading_dense) {
      DenseFfn(il, normed, ffn_out);
    } else {
      MoEFfn(il, session, normed, ffn_out);
    }
    for (std::uint32_t i = 0; i < hidden; ++i) {
      session.x[i] = ffn_in[i] + ffn_out[i];
    }
  }

  Rms(session.x, Load("output_norm.weight").w());
  // Stream the lm_head: materializing [hidden, vocab] is ~1.2 GiB of F32.
  {
    const auto& head = LoadRow("output.weight");
    std::vector<float> rowbuf(head.cols());
    for (std::uint64_t r = 0; r < head.rows(); ++r) {
      head.DequantizeRow(r, rowbuf.data());
      double acc = 0.0;
      for (std::uint64_t i = 0; i < head.cols(); ++i) {
        acc += static_cast<double>(rowbuf[i]) * session.x[i];
      }
      session.logits[r] = static_cast<float>(acc);
    }
  }
  session.position += 1;
  return session.logits;
}

}  // namespace gufo::models::flint
