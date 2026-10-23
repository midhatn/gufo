#include "src/models/glm53_flash/kernels/rocm/executor.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/kernels/rocm/kda.hpp"
#include "src/models/glm53_flash/kernels/rocm/matvec.hpp"
#include "src/models/glm53_flash/kernels/rocm/mhc.hpp"
#include "src/models/glm53_flash/kernels/rocm/moe.hpp"

extern "C" int ds4_mmq_init(int device);

namespace gufo::models::glm53_flash::rocm {

namespace {

using core::GgmlType;

constexpr std::size_t kTailMargin = 4096;
constexpr std::size_t kUploadChunk = std::size_t{128} << 20;
constexpr std::size_t kMatvecScratchBytes = 1 << 20;
constexpr int kHidden = kHcHidden;
constexpr int kStreams = kHcStreams;
constexpr int kDenseFf = 12288;
constexpr int kExpertFf = 2048;
constexpr int kQLora = 1536;
constexpr int kFfnScratch = std::max(kDenseFf, kMoeUsed* kExpertFf);
constexpr int kMlaQ = kDsaHeads * 256;
constexpr float kRmsEps = 1e-5F;
constexpr float kKNormEps = 1e-6F;

int TypeCode(GgmlType t) {
  switch (t) {
    case GgmlType::kF32:
      return kTypeF32;
    case GgmlType::kQ8_0:
      return kTypeQ8_0;
    case GgmlType::kQ6_K:
      return kTypeQ6_K;
    case GgmlType::kIQ2_XXS:
      return kTypeIQ2_XXS;
    case GgmlType::kIQ2_S:
      return kTypeIQ2_S;
    default:
      return -1;
  }
}

template<typename T>
bool Alloc(T** ptr, std::size_t count, std::string* error) {
  if (hipMalloc(reinterpret_cast<void**>(ptr), count * sizeof(T)) !=
      hipSuccess) {
    if (error != nullptr) {
      *error =
          "hipMalloc failed (" + std::to_string(count * sizeof(T)) + " bytes)";
    }
    *ptr = nullptr;
    return false;
  }
  if (hipMemset(*ptr, 0, count * sizeof(T)) != hipSuccess) {
    if (error != nullptr)
      *error = "hipMemset failed";
    return false;
  }
  return true;
}

std::string P(std::uint32_t layer, const char* name) {
  return "blk." + std::to_string(layer) + "." + name;
}

}  // namespace

struct Executor::LayerWeights {
  bool kda{false};
  bool dense{false};
  const DeviceTensor *hc_attn_fn{}, *hc_attn_scale{}, *hc_attn_base{},
      *attn_norm{}, *hc_ffn_fn{}, *hc_ffn_scale{}, *hc_ffn_base{}, *ffn_norm{};
  // KDA
  const DeviceTensor *attn_q{}, *attn_k{}, *attn_v{}, *conv_q{}, *conv_k{},
      *conv_v{}, *ssm_f_a{}, *ssm_f_b{}, *ssm_beta{}, *ssm_g_a{}, *ssm_g_b{},
      *ssm_dt{}, *ssm_a{}, *ssm_norm{}, *attn_output{};
  // DSA
  const DeviceTensor *q_a{}, *q_a_norm{}, *q_b{}, *kv_a{}, *kv_a_norm{}, *k_b{},
      *v_b{}, *idx_q_b{}, *idx_k{}, *idx_k_norm_w{}, *idx_k_norm_b{},
      *idx_gate{}, *idx_ape{}, *idx_proj{};
  // FFN
  const DeviceTensor *ffn_gate{}, *ffn_up{}, *ffn_down{};
  const DeviceTensor *router{}, *router_bias{}, *exp_gate{}, *exp_up{},
      *exp_down{}, *sh_gate{}, *sh_up{}, *sh_down{};
};

struct Executor::Scratch {
  void* matvec{nullptr};
  float* x[2]{nullptr, nullptr};
  float *y{}, *post{}, *comb{}, *block{}, *ffn_out{};
  // KDA
  float *kq{}, *kk{}, *kv{}, *decay{}, *beta{}, *fa{}, *fproj{}, *braw{}, *ga{},
      *g2{}, *o{};
  // DSA
  float *qr{}, *q{}, *lat{}, *iq{}, *ik{}, *gt{}, *wp{}, *scores{}, *qabs{},
      *ctx{}, *vout{};
  int* sel{};
  // FFN
  float *gate{}, *up{}, *act{};
  float *rlogits{}, *wts{}, *parts{}, *sg{}, *su{}, *sact{}, *shared{};
  int* ids{};
  // Head
  float *head_in{}, *logits{};
  std::vector<void*> all;
};

Executor::Session::~Session() {
  for (auto& l : layers) {
    for (void* p :
         {static_cast<void*>(l.conv), static_cast<void*>(l.state),
          static_cast<void*>(l.cache.lat), static_cast<void*>(l.cache.ik),
          static_cast<void*>(l.cache.gate),
          static_cast<void*>(l.cache.pooled)}) {
      if (p != nullptr)
        hipFree(p);
    }
  }
}

Executor::~Executor() {
  hipDeviceSynchronize();
  if (staging_ != nullptr)
    hipHostFree(staging_);
  for (auto& [name, t] : tensors_) {
    if (t.data != nullptr)
      hipFree(t.data);
  }
  if (scratch_) {
    for (void* p : scratch_->all) {
      hipFree(p);
    }
  }
}

// Streams a tensor to the device through a pinned staging buffer with large
// preads. Copying straight from the mapping faults the file in 128 KiB at a
// time, which both crawls and thrashes the page cache against the device
// allocations on a unified-memory machine.
bool Executor::CopyToDevice(void* dst, const core::GgufTensorInfo& t,
                            std::string* error) {
  const char* src = static_cast<const char*>(t.data);
  int fd = -1;
  std::uint64_t file_offset = 0;
  for (const auto& region : reader_->GetMappedRegions()) {
    const char* base = static_cast<const char*>(region.data);
    if (src >= base && src < base + region.size) {
      fd = region.file_descriptor;
      file_offset = static_cast<std::uint64_t>(src - base);
      break;
    }
  }
  if (fd < 0) {
    if (hipMemcpy(dst, src, t.size_bytes, hipMemcpyHostToDevice) !=
        hipSuccess) {
      if (error != nullptr)
        *error = "upload failed for " + std::string(t.name);
      return false;
    }
    return true;
  }
  if (staging_ == nullptr &&
      hipHostMalloc(&staging_, kUploadChunk, hipHostMallocDefault) !=
          hipSuccess) {
    if (error != nullptr)
      *error = "pinned staging allocation failed";
    return false;
  }
  for (std::size_t done = 0; done < t.size_bytes;) {
    const std::size_t n = std::min(kUploadChunk, t.size_bytes - done);
    std::size_t got = 0;
    while (got < n) {
      const ssize_t r = pread(fd, static_cast<char*>(staging_) + got, n - got,
                              static_cast<off_t>(file_offset + done + got));
      if (r <= 0) {
        if (error != nullptr)
          *error = "read failed for " + std::string(t.name);
        return false;
      }
      got += static_cast<std::size_t>(r);
    }
    if (hipMemcpy(static_cast<char*>(dst) + done, staging_, n,
                  hipMemcpyHostToDevice) != hipSuccess) {
      if (error != nullptr)
        *error = "upload failed for " + std::string(t.name);
      return false;
    }
    posix_fadvise(fd, static_cast<off_t>(file_offset + done),
                  static_cast<off_t>(n), POSIX_FADV_DONTNEED);
    done += n;
  }
  return true;
}

bool Executor::Upload(const std::string& name, std::string* error) {
  const auto* t = reader_->FindTensor(name);
  if (t == nullptr) {
    if (error != nullptr)
      *error = "missing tensor " + name;
    return false;
  }
  const int code = TypeCode(t->type);
  if (code < 0) {
    if (error != nullptr)
      *error = "unsupported tensor type for " + name;
    return false;
  }
  DeviceTensor d;
  d.type = code;
  d.k = static_cast<std::uint32_t>(t->dimensions[0]);
  d.rows = t->dimensions.size() > 1
               ? static_cast<std::uint32_t>(t->dimensions[1])
               : 1;
  d.experts = t->dimensions.size() > 2
                  ? static_cast<std::uint32_t>(t->dimensions[2])
                  : 1;
  if (hipMalloc(&d.data, t->size_bytes + kTailMargin) != hipSuccess) {
    if (error != nullptr)
      *error = "hipMalloc failed for " + name;
    return false;
  }
  if (!CopyToDevice(d.data, *t, error)) {
    hipFree(d.data);
    return false;
  }
  tensors_.emplace(name, d);
  return true;
}

const DeviceTensor* Executor::Find(const std::string& name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

std::unique_ptr<Executor> Executor::Create(const std::string& path,
                                           const ExecutorOptions& options,
                                           std::string* error) {
  std::unique_ptr<Executor> ex(new Executor());
  ex->reader_ = core::GgufReader::OpenFile(path, error);
  if (!ex->reader_)
    return nullptr;
  auto config = ParseConfig(*ex->reader_, error);
  if (!config)
    return nullptr;
  ex->config_ = *config;
  ex->options_ = options;
  ex->layers_ = options.layers == 0
                    ? ex->config_.TrunkLayers()
                    : std::min(options.layers, ex->config_.TrunkLayers());
  if (options.top_pools == 0 || options.top_pools > kDsaTopPools) {
    if (error != nullptr)
      *error = "top_pools out of range";
    return nullptr;
  }
  if (hipSetDevice(0) != hipSuccess) {
    if (error != nullptr)
      *error = "no HIP device";
    return nullptr;
  }
  if (ds4_mmq_init(0) != 0) {
    if (error != nullptr)
      *error = "mmq init failed";
    return nullptr;
  }
  const bool full = ex->layers_ == ex->config_.TrunkLayers();
  const bool head = options.head && full;
  ex->options_.head = head;
  for (const char* name : {"token_embd.weight"}) {
    if (ex->reader_->FindTensor(name) == nullptr) {
      if (error != nullptr)
        *error = std::string("missing tensor ") + name;
      return nullptr;
    }
  }
  if (head) {
    for (const char* name : {"output_norm.weight", "output.weight"}) {
      if (!ex->Upload(name, error))
        return nullptr;
    }
  }
  for (std::uint32_t layer = 0; layer < ex->layers_; ++layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    for (const auto& t : ex->reader_->GetTensors()) {
      if (t.name.substr(0, prefix.size()) == prefix) {
        if (!ex->Upload(std::string(t.name), error))
          return nullptr;
      }
    }
  }
  if (!ex->Resolve(error) || !ex->AllocScratch(error))
    return nullptr;
  return ex;
}

bool Executor::Resolve(std::string* error) {
  weights_.resize(layers_);
  for (std::uint32_t l = 0; l < layers_; ++l) {
    LayerWeights& w = weights_[l];
    w.kda = config_.IsKda(l);
    w.dense = config_.IsDense(l);
    bool ok = true;
    const auto get = [&](const DeviceTensor*& slot, const char* name) {
      slot = Find(P(l, name));
      if (slot == nullptr) {
        ok = false;
        if (error != nullptr && error->empty()) {
          *error = "missing tensor " + P(l, name);
        }
      }
    };
    get(w.hc_attn_fn, "hc_attn_fn.weight");
    get(w.hc_attn_scale, "hc_attn_scale.weight");
    get(w.hc_attn_base, "hc_attn_base.weight");
    get(w.attn_norm, "attn_norm.weight");
    get(w.hc_ffn_fn, "hc_ffn_fn.weight");
    get(w.hc_ffn_scale, "hc_ffn_scale.weight");
    get(w.hc_ffn_base, "hc_ffn_base.weight");
    get(w.ffn_norm, "ffn_norm.weight");
    get(w.attn_output, "attn_output.weight");
    if (w.kda) {
      get(w.attn_q, "attn_q.weight");
      get(w.attn_k, "attn_k.weight");
      get(w.attn_v, "attn_v.weight");
      get(w.conv_q, "ssm_conv1d_q.weight");
      get(w.conv_k, "ssm_conv1d_k.weight");
      get(w.conv_v, "ssm_conv1d_v.weight");
      get(w.ssm_f_a, "ssm_f_a.weight");
      get(w.ssm_f_b, "ssm_f_b.weight");
      get(w.ssm_beta, "ssm_beta.weight");
      get(w.ssm_g_a, "ssm_g_a.weight");
      get(w.ssm_g_b, "ssm_g_b.weight");
      get(w.ssm_dt, "ssm_dt.bias");
      get(w.ssm_a, "ssm_a");
      get(w.ssm_norm, "ssm_norm.weight");
    } else {
      get(w.q_a, "attn_q_a.weight");
      get(w.q_a_norm, "attn_q_a_norm.weight");
      get(w.q_b, "attn_q_b.weight");
      get(w.kv_a, "attn_kv_a_mqa.weight");
      get(w.kv_a_norm, "attn_kv_a_norm.weight");
      get(w.k_b, "attn_k_b.weight");
      get(w.v_b, "attn_v_b.weight");
      get(w.idx_q_b, "indexer.attn_q_b.weight");
      get(w.idx_k, "indexer.attn_k.weight");
      get(w.idx_k_norm_w, "indexer.k_norm.weight");
      get(w.idx_k_norm_b, "indexer.k_norm.bias");
      get(w.idx_gate, "indexer_compressor_gate.weight");
      get(w.idx_ape, "indexer_compressor_ape.weight");
      get(w.idx_proj, "indexer.proj.weight");
    }
    if (w.dense) {
      get(w.ffn_gate, "ffn_gate.weight");
      get(w.ffn_up, "ffn_up.weight");
      get(w.ffn_down, "ffn_down.weight");
    } else {
      get(w.router, "ffn_gate_inp.weight");
      get(w.router_bias, "exp_probs_b.bias");
      get(w.exp_gate, "ffn_gate_exps.weight");
      get(w.exp_up, "ffn_up_exps.weight");
      get(w.exp_down, "ffn_down_exps.weight");
      get(w.sh_gate, "ffn_gate_shexp.weight");
      get(w.sh_up, "ffn_up_shexp.weight");
      get(w.sh_down, "ffn_down_shexp.weight");
    }
    if (!ok)
      return false;
  }
  return true;
}

bool Executor::AllocScratch(std::string* error) {
  scratch_ = std::make_unique<Scratch>();
  Scratch& s = *scratch_;
  const std::size_t c = options_.prefill_chunk;
  const std::size_t pools = options_.max_context / kDsaPool + 1;
  bool ok = true;
  const auto take = [&](auto** ptr, std::size_t count) {
    if (!ok)
      return;
    ok = Alloc(ptr, count, error);
    if (ok)
      s.all.push_back(static_cast<void*>(*ptr));
  };
  take(reinterpret_cast<char**>(&s.matvec), kMatvecScratchBytes);
  take(&s.x[0], c * kStreams * kHidden);
  take(&s.x[1], c * kStreams * kHidden);
  take(&s.y, c * kHidden);
  take(&s.post, c * kStreams);
  take(&s.comb, c * kStreams * kStreams);
  take(&s.block, c * kHidden);
  take(&s.ffn_out, c * kHidden);
  take(&s.kq, c * kKdaInner);
  take(&s.kk, c * kKdaInner);
  take(&s.kv, c * kKdaInner);
  take(&s.decay, c * kKdaInner);
  take(&s.beta, c * kKdaHeads);
  take(&s.fa, c * kKdaHeadDim);
  take(&s.fproj, c * kKdaInner);
  take(&s.braw, c * kKdaHeads);
  take(&s.ga, c * kKdaHeadDim);
  take(&s.g2, c * kKdaInner);
  take(&s.o, c * kKdaInner);
  take(&s.qr, c * kQLora);
  take(&s.q, c * kMlaQ);
  take(&s.lat, c * kDsaLatent);
  take(&s.iq, c * kDsaIndexHeads * kDsaIndexDim);
  take(&s.ik, c * kDsaIndexDim);
  take(&s.gt, c * kDsaIndexDim);
  take(&s.wp, c * kDsaIndexHeads);
  take(&s.scores, pools);
  take(&s.qabs, kDsaHeads * kDsaLatent);
  take(&s.ctx, kDsaHeads * kDsaLatent);
  take(&s.vout, c * kMlaQ);
  take(&s.sel, kDsaTopPools);
  take(&s.gate, c * kFfnScratch);
  take(&s.up, c * kFfnScratch);
  take(&s.act, c * kFfnScratch);
  take(&s.rlogits, c * kMoeExperts);
  take(&s.wts, c * kMoeUsed);
  take(&s.ids, c * kMoeUsed);
  take(&s.parts, c * kMoeUsed * kHidden);
  take(&s.sg, c * kExpertFf);
  take(&s.su, c * kExpertFf);
  take(&s.sact, c * kExpertFf);
  take(&s.shared, c * kHidden);
  take(&s.head_in, kHidden);
  take(&s.logits, config_.vocab_size);
  return ok;
}

std::unique_ptr<Executor::Session> Executor::CreateSession(std::string* error) {
  auto s = std::make_unique<Session>();
  s->layers.resize(layers_);
  bool ok = true;
  const std::size_t ctx = options_.max_context;
  for (std::uint32_t l = 0; ok && l < layers_; ++l) {
    LayerState& st = s->layers[l];
    if (weights_[l].kda) {
      ok =
          Alloc(&st.conv, 3 * 3 * kKdaInner, error) &&
          Alloc(&st.state,
                static_cast<std::size_t>(kKdaHeads) * kKdaHeadDim * kKdaHeadDim,
                error);
    } else {
      ok = Alloc(&st.cache.lat, ctx * kDsaLatent, error) &&
           Alloc(&st.cache.ik, ctx * kDsaIndexDim, error) &&
           Alloc(&st.cache.gate, ctx * kDsaIndexDim, error) &&
           Alloc(&st.cache.pooled, (ctx / kDsaPool + 1) * kDsaIndexDim, error);
    }
  }
  return ok ? std::move(s) : nullptr;
}

void Executor::Proj(const DeviceTensor& w, const float* x, float* out,
                    int tokens) const {
  Gemm(w.type, w.data, x, out, static_cast<int>(w.rows), tokens,
       static_cast<int>(w.k), scratch_->matvec, stream_);
}

void Executor::KdaBlock(std::uint32_t layer, Session& s, int T) const {
  const LayerWeights& w = weights_[layer];
  const Scratch& b = *scratch_;
  LayerState& st = s.layers[layer];
  Proj(*w.attn_q, b.y, b.kq, T);
  Proj(*w.attn_k, b.y, b.kk, T);
  Proj(*w.attn_v, b.y, b.kv, T);
  KdaConv(b.kq, b.kk, b.kv, st.conv, w.conv_q->f32(), w.conv_k->f32(),
          w.conv_v->f32(), stream_, T);
  Proj(*w.ssm_f_a, b.y, b.fa, T);
  Proj(*w.ssm_f_b, b.fa, b.fproj, T);
  Proj(*w.ssm_beta, b.y, b.braw, T);
  KdaGates(b.fproj, w.ssm_dt->f32(), w.ssm_a->f32(), b.braw, b.decay, b.beta,
           stream_, T);
  KdaRecur(b.kq, b.kk, b.kv, b.decay, b.beta, st.state, b.o, stream_, T);
  Proj(*w.ssm_g_a, b.y, b.ga, T);
  Proj(*w.ssm_g_b, b.ga, b.g2, T);
  KdaOut(b.o, w.ssm_norm->f32(), b.g2, stream_, T);
  Proj(*w.attn_output, b.o, b.block, T);
}

void Executor::DsaBlock(std::uint32_t layer, Session& s, int T) const {
  const LayerWeights& w = weights_[layer];
  const Scratch& b = *scratch_;
  LayerState& st = s.layers[layer];
  Proj(*w.q_a, b.y, b.qr, T);
  RmsNorm(b.qr, w.q_a_norm->f32(), kQLora, kRmsEps, stream_, T);
  Proj(*w.q_b, b.qr, b.q, T);
  Proj(*w.kv_a, b.y, b.lat, T);
  RmsNorm(b.lat, w.kv_a_norm->f32(), kDsaLatent, kRmsEps, stream_, T);
  Proj(*w.idx_q_b, b.qr, b.iq, T);
  Proj(*w.idx_k, b.y, b.ik, T);
  LayerNorm(b.ik, w.idx_k_norm_w->f32(), w.idx_k_norm_b->f32(), kDsaIndexDim,
            kKNormEps, stream_, T);
  Proj(*w.idx_gate, b.y, b.gt, T);
  Proj(*w.idx_proj, b.y, b.wp, T);
  for (int t = 0; t < T; ++t) {
    const int pos = static_cast<int>(s.position) + t;
    DsaAppend(st.cache, b.lat + static_cast<std::size_t>(t) * kDsaLatent,
              b.ik + static_cast<std::size_t>(t) * kDsaIndexDim,
              b.gt + static_cast<std::size_t>(t) * kDsaIndexDim,
              w.idx_ape->f32(), pos, stream_);
    const int npool = (pos + 1) / kDsaPool;
    const int nsel = std::min<int>(npool, options_.top_pools);
    DsaScores(
        b.iq + static_cast<std::size_t>(t) * kDsaIndexHeads * kDsaIndexDim,
        b.wp + static_cast<std::size_t>(t) * kDsaIndexHeads, st.cache.pooled,
        npool, b.scores, stream_);
    DsaSelect(b.scores, npool, static_cast<int>(options_.top_pools), b.sel,
              stream_);
    MatvecBatched(w.k_b->type, w.k_b->data,
                  b.q + static_cast<std::size_t>(t) * kMlaQ, b.qabs, kDsaLatent,
                  256, kDsaHeads, b.matvec, stream_);
    DsaAttend(b.qabs, st.cache.lat, b.sel, nsel, pos, 1.0F / 16.0F, b.ctx,
              stream_);
    MatvecBatched(w.v_b->type, w.v_b->data, b.ctx,
                  b.vout + static_cast<std::size_t>(t) * kMlaQ, 256, kDsaLatent,
                  kDsaHeads, b.matvec, stream_);
  }
  Proj(*w.attn_output, b.vout, b.block, T);
}

void Executor::FfnBlock(std::uint32_t layer, int T) const {
  const LayerWeights& w = weights_[layer];
  const Scratch& b = *scratch_;
  if (w.dense) {
    Proj(*w.ffn_gate, b.y, b.gate, T);
    Proj(*w.ffn_up, b.y, b.up, T);
    SwigluClamp(b.gate, b.up, b.act, T * kDenseFf, stream_);
    Proj(*w.ffn_down, b.act, b.ffn_out, T);
    return;
  }
  Proj(*w.router, b.y, b.rlogits, T);
  MoeRoute(b.rlogits, w.router_bias->f32(), config_.expert_weights_scale, b.ids,
           b.wts, stream_, T);
  if (T == 1) {
    MatvecExperts(w.exp_gate->type, w.exp_gate->data, b.y, b.gate, kExpertFf,
                  kHidden, kMoeExperts, kMoeUsed, b.ids, false, b.matvec,
                  stream_);
    MatvecExperts(w.exp_up->type, w.exp_up->data, b.y, b.up, kExpertFf, kHidden,
                  kMoeExperts, kMoeUsed, b.ids, false, b.matvec, stream_);
    SwigluClamp(b.gate, b.up, b.act, kMoeUsed * kExpertFf, stream_);
    MatvecExperts(w.exp_down->type, w.exp_down->data, b.act, b.parts, kHidden,
                  kExpertFf, kMoeExperts, kMoeUsed, b.ids, true, b.matvec,
                  stream_);
  } else {
    ExpertsGemm(w.exp_gate->type, w.exp_gate->data, b.y, b.gate, kExpertFf,
                kHidden, T, kMoeExperts, kMoeUsed, b.ids, stream_);
    ExpertsGemm(w.exp_up->type, w.exp_up->data, b.y, b.up, kExpertFf, kHidden,
                T, kMoeExperts, kMoeUsed, b.ids, stream_);
    SwigluClamp(b.gate, b.up, b.act, T * kMoeUsed * kExpertFf, stream_);
    ExpertsGemm(w.exp_down->type, w.exp_down->data, b.act, b.parts, kHidden,
                kExpertFf, T * kMoeUsed, kMoeExperts, 1, b.ids, stream_);
  }
  Proj(*w.sh_gate, b.y, b.sg, T);
  Proj(*w.sh_up, b.y, b.su, T);
  SwigluClamp(b.sg, b.su, b.sact, T * kExpertFf, stream_);
  Proj(*w.sh_down, b.sact, b.shared, T);
  MoeCombine(b.parts, b.wts, b.shared, b.ffn_out, stream_, T);
}

bool Executor::Forward(Session& session, const std::int32_t* tokens, int T,
                       bool want_logits, std::string* error) {
  if (T <= 0 || T > static_cast<int>(options_.prefill_chunk)) {
    if (error != nullptr)
      *error = "bad chunk size";
    return false;
  }
  if (session.position + static_cast<std::uint32_t>(T) > options_.max_context) {
    if (error != nullptr)
      *error = "context full";
    return false;
  }
  const auto* embd = reader_->FindTensor("token_embd.weight");
  std::vector<float> host(static_cast<std::size_t>(T) * kStreams * kHidden);
  const std::size_t row_bytes = embd->size_bytes / embd->dimensions[1];
  for (int t = 0; t < T; ++t) {
    const std::int32_t token = tokens[t];
    if (token < 0 || static_cast<std::uint64_t>(token) >= embd->dimensions[1]) {
      if (error != nullptr)
        *error = "token out of range";
      return false;
    }
    float* row = host.data() + static_cast<std::size_t>(t) * kStreams * kHidden;
    const char* src = static_cast<const char*>(embd->data) +
                      static_cast<std::size_t>(token) * row_bytes;
    switch (embd->type) {
      case GgmlType::kQ6_K:
        quant::DequantizeQ6_K(src, row, kHidden);
        break;
      case GgmlType::kQ8_0:
        quant::DequantizeQ8_0(src, row, kHidden);
        break;
      case GgmlType::kF32:
        std::memcpy(row, src, kHidden * sizeof(float));
        break;
      default:
        if (error != nullptr)
          *error = "unsupported embedding type";
        return false;
    }
    for (int s = 1; s < kStreams; ++s) {
      std::memcpy(row + s * kHidden, row, kHidden * sizeof(float));
    }
  }
  Scratch& b = *scratch_;
  int cur = 0;
  hipMemcpyAsync(b.x[cur], host.data(), host.size() * sizeof(float),
                 hipMemcpyHostToDevice, stream_);
  hipStreamSynchronize(stream_);

  for (std::uint32_t layer = 0; layer < layers_; ++layer) {
    const LayerWeights& w = weights_[layer];
    HcPre(b.x[cur], w.hc_attn_fn->f32(), w.hc_attn_scale->f32(),
          w.hc_attn_base->f32(), w.attn_norm->f32(), config_.hc_eps, b.y,
          b.post, b.comb, stream_, T);
    if (w.kda) {
      KdaBlock(layer, session, T);
    } else {
      DsaBlock(layer, session, T);
    }
    HcPost(b.x[cur], b.block, b.post, b.comb, b.x[1 - cur], stream_, T);
    cur = 1 - cur;

    HcPre(b.x[cur], w.hc_ffn_fn->f32(), w.hc_ffn_scale->f32(),
          w.hc_ffn_base->f32(), w.ffn_norm->f32(), config_.hc_eps, b.y, b.post,
          b.comb, stream_, T);
    FfnBlock(layer, T);
    HcPost(b.x[cur], b.ffn_out, b.post, b.comb, b.x[1 - cur], stream_, T);
    cur = 1 - cur;
  }
  last_residual_ =
      b.x[cur] + static_cast<std::size_t>(T - 1) * kStreams * kHidden;
  if (want_logits && options_.head) {
    const DeviceTensor& norm = *Find("output_norm.weight");
    const DeviceTensor& out = *Find("output.weight");
    HeadNorm(last_residual_, norm.f32(), b.head_in, stream_);
    Proj(out, b.head_in, b.logits, 1);
  }
  session.position += static_cast<std::uint32_t>(T);
  if (hipStreamSynchronize(stream_) != hipSuccess ||
      hipGetLastError() != hipSuccess) {
    if (error != nullptr)
      *error = "HIP error during forward";
    return false;
  }
  return true;
}

bool Executor::Evaluate(Session& session, std::int32_t token,
                        std::string* error) {
  return Forward(session, &token, 1, true, error);
}

bool Executor::Prefill(Session& session, std::span<const std::int32_t> tokens,
                       std::string* error) {
  if (tokens.empty()) {
    if (error != nullptr)
      *error = "empty prompt";
    return false;
  }
  const std::size_t chunk = options_.prefill_chunk;
  for (std::size_t at = 0; at < tokens.size(); at += chunk) {
    const std::size_t n = std::min(chunk, tokens.size() - at);
    const bool last = at + n == tokens.size();
    if (!Forward(session, tokens.data() + at, static_cast<int>(n), last,
                 error)) {
      return false;
    }
  }
  return true;
}

std::vector<float> Executor::Logits(const Session&) {
  std::vector<float> out(config_.vocab_size);
  hipMemcpy(out.data(), scratch_->logits, out.size() * sizeof(float),
            hipMemcpyDeviceToHost);
  return out;
}

std::vector<float> Executor::Residual(const Session&) {
  std::vector<float> out(kStreams * kHidden);
  hipMemcpy(out.data(), last_residual_, out.size() * sizeof(float),
            hipMemcpyDeviceToHost);
  return out;
}

}  // namespace gufo::models::glm53_flash::rocm
