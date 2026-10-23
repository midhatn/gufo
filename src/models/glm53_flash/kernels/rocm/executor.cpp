#include "src/models/glm53_flash/kernels/rocm/executor.hpp"

#include <fcntl.h>
#include <hipblas/hipblas.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/kernels/rocm/kda.hpp"
#include "src/models/glm53_flash/kernels/rocm/matvec.hpp"
#include "src/models/glm53_flash/kernels/rocm/mhc.hpp"
#include "src/models/glm53_flash/kernels/rocm/moe.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

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
constexpr std::uint32_t kPairTileRows = 128;
constexpr std::uint32_t kDownTileRows = 64;
constexpr std::uint32_t kPadRows = 16;
// Batches at least this wide use the W8A8 GEMMs.
constexpr int kWideBatch = 32;
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
      *ctx{}, *vout{}, *q_heads{}, *v_heads{};
  void* head_half{};
  int* sel{};
  // FFN
  float *gate{}, *up{}, *act{};
  float *rlogits{}, *wts{}, *parts{}, *sg{}, *su{}, *sact{}, *shared{};
  int* ids{};
  // Head
  float *head_in{}, *logits{}, *dflash_capture{};
  int* argmax{};
  int* pos_dev{};
  char* q8_arena{};
  std::size_t q8_arena_bytes{0};
  // Routed-expert F16 route (prefill).
  void *x_half{}, *act_half{}, *parts_half{};
  std::uint32_t* counts{};
  std::int32_t *bounds{}, *cursors{}, *rows_token{}, *rows_slot{},
      *tiles_pair{}, *tiles_down{};
  std::uint32_t* host_counts{};
  std::int32_t* host_tiles{};
  std::uint32_t tile_capacity{0};
  long long score_stride{0};
  std::vector<void*> all;
};

Executor::Session::~Session() {
  for (auto& log : kda_log) {
    for (void* p :
         {static_cast<void*>(log.raw_q), static_cast<void*>(log.raw_k),
          static_cast<void*>(log.raw_v), static_cast<void*>(log.decay),
          static_cast<void*>(log.beta)}) {
      if (p != nullptr)
        hipFree(p);
    }
  }
  for (float* p : saved_conv) {
    if (p != nullptr)
      hipFree(p);
  }
  for (float* p : saved_state) {
    if (p != nullptr)
      hipFree(p);
  }
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
  DropGraph();
  if (capture_stream_ != nullptr)
    hipStreamDestroy(capture_stream_);
  if (staging_ != nullptr)
    hipHostFree(staging_);
  if (blas_ != nullptr)
    hipblasDestroy(static_cast<hipblasHandle_t>(blas_));
  for (auto& [name, t] : tensors_) {
    if (t.q8_owned && t.q8 != nullptr)
      hipFree(t.q8);
    if (t.f16 != nullptr)
      hipFree(t.f16);
    if (t.data != nullptr)
      hipFree(t.data);
  }
  if (scratch_) {
    for (void* p : scratch_->all) {
      hipFree(p);
    }
    if (scratch_->host_counts != nullptr)
      hipHostFree(scratch_->host_counts);
    if (scratch_->host_tiles != nullptr)
      hipHostFree(scratch_->host_tiles);
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
  d.slab_bytes = t->size_bytes / d.experts;
  if (hipMalloc(&d.data, t->size_bytes + kTailMargin) != hipSuccess) {
    if (error != nullptr)
      *error = "hipMalloc failed for " + name;
    return false;
  }
  if (!CopyToDevice(d.data, *t, error)) {
    hipFree(d.data);
    return false;
  }
  if (d.experts == 1 && name.rfind("blk.", 0) == 0 && d.k % 32 == 0) {
    if (d.type == kTypeQ8_0) {
      d.q8 = d.data;
    } else if (d.type == kTypeQ6_K) {
      const std::size_t bytes =
          static_cast<std::size_t>(d.rows) * (d.k / 32) * 34;
      if (hipMalloc(&d.q8, bytes + kTailMargin) == hipSuccess) {
        d.q8_owned = true;
        gufo::models::qwen38_flash_next::rocm::RequantToQ8_0(
            d.data, static_cast<std::uint32_t>(GgmlType::kQ6_K), d.k, d.rows,
            d.q8, stream_);
        hipStreamSynchronize(stream_);
      } else {
        d.q8 = nullptr;
      }
    }
  }
  if (d.experts > 1 && d.type == kTypeQ6_K &&
      (name.find("attn_k_b") != std::string::npos ||
       name.find("attn_v_b") != std::string::npos)) {
    const std::size_t rows_total = static_cast<std::size_t>(d.rows) * d.experts;
    std::vector<__half> half(rows_total * d.k);
    std::vector<float> row(d.k);
    const std::size_t row_bytes = t->size_bytes / rows_total;
    for (std::size_t r = 0; r < rows_total; ++r) {
      quant::DequantizeQ6_K(static_cast<const char*>(t->data) + r * row_bytes,
                            row.data(), d.k);
      for (std::uint32_t i = 0; i < d.k; ++i) {
        half[r * d.k + i] = __float2half(row[i]);
      }
    }
    if (hipMalloc(&d.f16, half.size() * sizeof(__half)) == hipSuccess) {
      hipMemcpy(d.f16, half.data(), half.size() * sizeof(__half),
                hipMemcpyHostToDevice);
    }
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
  ex->profile_ = std::getenv("GUFO_GLM53_PROFILE") != nullptr;
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
  {
    hipblasHandle_t handle = nullptr;
    if (hipblasCreate(&handle) != HIPBLAS_STATUS_SUCCESS) {
      if (error != nullptr)
        *error = "hipblas init failed";
      return nullptr;
    }
    ex->blas_ = handle;
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
  if (!options.dflash_layers.empty()) {
    if (options.dflash_rows == 0) {
      if (error != nullptr)
        *error = "drafter taps need dflash_rows";
      return nullptr;
    }
    for (const auto tap : options.dflash_layers) {
      if (tap >= ex->layers_) {
        if (error != nullptr)
          *error = "drafter tap layer exceeds the loaded trunk";
        return nullptr;
      }
    }
    ex->dflash_taps_ = options.dflash_layers;
    if (!ex->Upload("token_embd.weight", error))
      return nullptr;
    ex->token_embd_dev_ = ex->tensors_.at("token_embd.weight");
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
  s.score_stride = static_cast<long long>(pools);
  take(&s.scores, c * pools);
  take(&s.qabs, c * kDsaHeads * kDsaLatent);
  take(&s.ctx, c * kDsaHeads * kDsaLatent);
  take(reinterpret_cast<std::uint16_t**>(&s.head_half),
       c * kDsaHeads * kDsaLatent);
  take(&s.q_heads, c * kMlaQ);
  take(&s.v_heads, c * kMlaQ);
  take(&s.vout, c * kMlaQ);
  take(&s.sel, c * kDsaTopPools);
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
  const std::size_t slots = c * kMoeUsed;
  const std::size_t compact_rows =
      gufo::models::qwen38_flash_next::rocm::RoutedCompactRows(slots,
                                                               kMoeExperts);
  s.tile_capacity = static_cast<std::uint32_t>(
      (slots + std::size_t{kMoeExperts} * (kPadRows - 1)) / kPadRows +
      kMoeExperts + 1);
  take(reinterpret_cast<std::uint16_t**>(&s.x_half), c * kHidden);
  take(reinterpret_cast<std::uint16_t**>(&s.act_half), slots * kExpertFf);
  take(reinterpret_cast<std::uint16_t**>(&s.parts_half), slots * kHidden);
  take(&s.counts, kMoeExperts);
  take(&s.bounds, kMoeExperts + 1);
  take(&s.cursors, kMoeExperts);
  take(&s.rows_token, compact_rows);
  take(&s.rows_slot, compact_rows);
  take(&s.tiles_pair, s.tile_capacity);
  take(&s.tiles_down, s.tile_capacity);
  if (ok && (hipHostMalloc(reinterpret_cast<void**>(&s.host_counts),
                           kMoeExperts * sizeof(std::uint32_t)) != hipSuccess ||
             hipHostMalloc(reinterpret_cast<void**>(&s.host_tiles),
                           2 * s.tile_capacity * sizeof(std::int32_t)) !=
                 hipSuccess)) {
    ok = false;
    if (error != nullptr)
      *error = "pinned routing buffers failed";
  }
  s.q8_arena_bytes = c * std::size_t{160} * 1024;
  take(&s.q8_arena, s.q8_arena_bytes);
  take(&s.head_in, std::size_t{options_.logit_rows} * kHidden);
  take(&s.argmax, options_.logit_rows);
  take(&s.pos_dev, 1);
  if (options_.dflash_rows != 0) {
    take(&s.dflash_capture, std::size_t{options_.dflash_rows} *
                                options_.dflash_layers.size() * kHidden);
  }
  take(&s.logits, std::size_t{options_.logit_rows} * config_.vocab_size);
  return ok;
}

std::unique_ptr<Executor::Session> Executor::CreateSession(std::string* error) {
  auto s = std::make_unique<Session>();
  s->id = ++session_counter_;
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

Executor::Act Executor::Prep(const float* x, int tokens, int k) const {
  if (tokens < kWideBatch) {
    return {x, nullptr};
  }
  namespace qfn = gufo::models::qwen38_flash_next::rocm;
  const std::size_t bytes = (qfn::Q8TiledBytes(tokens, k) + 255) / 256 * 256;
  if (arena_used_ + bytes > scratch_->q8_arena_bytes) {
    return {x, nullptr};
  }
  char* at = scratch_->q8_arena + arena_used_;
  arena_used_ += bytes;
  qfn::QuantizeQ8Tiled(x, at, tokens, k, stream_);
  return {x, at};
}

void Executor::Proj(const DeviceTensor& w, const Act& x, float* out,
                    int tokens) const {
  namespace qfn = gufo::models::qwen38_flash_next::rocm;
  if (x.q != nullptr && w.q8 != nullptr &&
      qfn::W8A8Gemm(w.q8, x.q, out, tokens, w.rows, w.k, stream_)) {
    return;
  }
  Gemm(w.type, w.data, x.f, out, static_cast<int>(w.rows), tokens,
       static_cast<int>(w.k), scratch_->matvec, stream_);
}

// out[h][t][r] = sum_k W_h[r][k] * in[h][t][k] for the 64 head slabs of a
// stacked [64][rows][k] tensor: one strided-batched GEMM over F16 copies of
// the slabs and the F16 activations.
void Executor::HeadGemm(const DeviceTensor& w, const float* in, float* out,
                        int T, int k, int rows) const {
  const Scratch& b = *scratch_;
  ToHalf(in, b.head_half, static_cast<long long>(kDsaHeads) * T * k, stream_);
  const auto handle = static_cast<hipblasHandle_t>(blas_);
  hipblasSetStream(handle, stream_);
  const float alpha = 1.0F;
  const float beta = 0.0F;
  hipblasGemmStridedBatchedEx(
      handle, HIPBLAS_OP_T, HIPBLAS_OP_N, rows, T, k, &alpha, w.f16,
      HIPBLAS_R_16F, k, static_cast<long long>(rows) * k, b.head_half,
      HIPBLAS_R_16F, k, static_cast<long long>(T) * k, &beta, out,
      HIPBLAS_R_32F, rows, static_cast<long long>(T) * rows, kDsaHeads,
      HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
}

void Executor::KdaBlock(std::uint32_t layer, Session& s, int T) const {
  const LayerWeights& w = weights_[layer];
  const Scratch& b = *scratch_;
  LayerState& st = s.layers[layer];
  arena_used_ = 0;
  const Act y = Prep(b.y, T, kHidden);
  Proj(*w.attn_q, y, b.kq, T);
  Proj(*w.attn_k, y, b.kk, T);
  Proj(*w.attn_v, y, b.kv, T);
  const bool record = recording_ && !s.kda_log.empty() &&
                      T <= static_cast<int>(options_.logit_rows);
  if (record) {
    const std::size_t bytes =
        static_cast<std::size_t>(T) * kKdaInner * sizeof(float);
    hipMemcpyAsync(s.kda_log[layer].raw_q, b.kq, bytes, hipMemcpyDeviceToDevice,
                   stream_);
    hipMemcpyAsync(s.kda_log[layer].raw_k, b.kk, bytes, hipMemcpyDeviceToDevice,
                   stream_);
    hipMemcpyAsync(s.kda_log[layer].raw_v, b.kv, bytes, hipMemcpyDeviceToDevice,
                   stream_);
  }
  KdaConv(b.kq, b.kk, b.kv, st.conv, w.conv_q->f32(), w.conv_k->f32(),
          w.conv_v->f32(), stream_, T);
  Proj(*w.ssm_f_a, y, b.fa, T);
  Proj(*w.ssm_f_b, Prep(b.fa, T, kKdaHeadDim), b.fproj, T);
  Proj(*w.ssm_beta, y, b.braw, T);
  KdaGates(b.fproj, w.ssm_dt->f32(), w.ssm_a->f32(), b.braw, b.decay, b.beta,
           stream_, T);
  if (record) {
    hipMemcpyAsync(s.kda_log[layer].decay, b.decay,
                   static_cast<std::size_t>(T) * kKdaInner * sizeof(float),
                   hipMemcpyDeviceToDevice, stream_);
    hipMemcpyAsync(s.kda_log[layer].beta, b.beta,
                   static_cast<std::size_t>(T) * kKdaHeads * sizeof(float),
                   hipMemcpyDeviceToDevice, stream_);
  }
  KdaRecur(b.kq, b.kk, b.kv, b.decay, b.beta, st.state, b.o, stream_, T);
  Proj(*w.ssm_g_a, y, b.ga, T);
  Proj(*w.ssm_g_b, Prep(b.ga, T, kKdaHeadDim), b.g2, T);
  KdaOut(b.o, w.ssm_norm->f32(), b.g2, stream_, T);
  Proj(*w.attn_output, Prep(b.o, T, kKdaInner), b.block, T);
}

void Executor::DsaBlock(std::uint32_t layer, Session& s, int T) const {
  const LayerWeights& w = weights_[layer];
  const Scratch& b = *scratch_;
  LayerState& st = s.layers[layer];
  arena_used_ = 0;
  const Act y = Prep(b.y, T, kHidden);
  Proj(*w.q_a, y, b.qr, T);
  RmsNorm(b.qr, w.q_a_norm->f32(), kQLora, kRmsEps, stream_, T);
  const Act qr = Prep(b.qr, T, kQLora);
  Proj(*w.q_b, qr, b.q, T);
  Proj(*w.kv_a, y, b.lat, T);
  RmsNorm(b.lat, w.kv_a_norm->f32(), kDsaLatent, kRmsEps, stream_, T);
  Proj(*w.idx_q_b, qr, b.iq, T);
  Proj(*w.idx_k, y, b.ik, T);
  LayerNorm(b.ik, w.idx_k_norm_w->f32(), w.idx_k_norm_b->f32(), kDsaIndexDim,
            kKNormEps, stream_, T);
  Proj(*w.idx_gate, y, b.gt, T);
  Proj(*w.idx_proj, y, b.wp, T);
  Mark(5);
  if (T >= kWideBatch) {
    const int pos0 = static_cast<int>(s.position);
    const int top = static_cast<int>(options_.top_pools);
    DsaAppendBatch(st.cache, b.lat, b.ik, b.gt, w.idx_ape->f32(), pos0, T,
                   stream_);
    DsaScoresBatch(b.iq, b.wp, st.cache.pooled, pos0, T, b.score_stride,
                   b.scores, stream_);
    DsaSelectBatch(b.scores, b.score_stride, pos0, T, top, b.sel, stream_);
    Mark(6);
    // attn_k_b / attn_v_b are 64 stacked per-head slabs: one GEMM per head
    // over all tokens, in the head-major layout the attention kernel reads.
    HeadMajor(b.q, b.q_heads, T, kDsaHeads, 256, stream_);
    HeadGemm(*w.k_b, b.q_heads, b.qabs, T, 256, kDsaLatent);
    Mark(7);
    ToHalf(b.qabs, b.head_half,
           static_cast<long long>(kDsaHeads) * T * kDsaLatent, stream_);
    DsaAttendBatch(b.head_half, st.cache.lat, b.sel, pos0, T, top, 1.0F / 16.0F,
                   b.ctx, stream_);
    Mark(8);
    HeadGemm(*w.v_b, b.ctx, b.v_heads, T, kDsaLatent, 256);
    TokenMajor(b.v_heads, b.vout, T, kDsaHeads, 256, stream_);
    Mark(7);
  } else if (T == 1) {
    const int top = static_cast<int>(options_.top_pools);
    DsaAppendDev(st.cache, b.lat, b.ik, b.gt, w.idx_ape->f32(), b.pos_dev,
                 stream_);
    DsaScoresDev(b.iq, b.wp, st.cache.pooled,
                 static_cast<int>(options_.max_context / kDsaPool), b.scores,
                 b.pos_dev, stream_);
    DsaSelectDev(b.scores, top, b.sel, b.pos_dev, stream_);
    MatvecBatched(w.k_b->type, w.k_b->data, b.q, b.qabs, kDsaLatent, 256,
                  kDsaHeads, b.matvec, stream_);
    DsaAttendDev(b.qabs, st.cache.lat, b.sel, top, 1.0F / 16.0F, b.ctx,
                 b.pos_dev, stream_);
    MatvecBatched(w.v_b->type, w.v_b->data, b.ctx, b.vout, 256, kDsaLatent,
                  kDsaHeads, b.matvec, stream_);
  } else {
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
                    b.q + static_cast<std::size_t>(t) * kMlaQ, b.qabs,
                    kDsaLatent, 256, kDsaHeads, b.matvec, stream_);
      DsaAttend(b.qabs, st.cache.lat, b.sel, nsel, pos, 1.0F / 16.0F, b.ctx,
                stream_);
      MatvecBatched(w.v_b->type, w.v_b->data, b.ctx,
                    b.vout + static_cast<std::size_t>(t) * kMlaQ, 256,
                    kDsaLatent, kDsaHeads, b.matvec, stream_);
    }
  }
  Proj(*w.attn_output, Prep(b.vout, T, kMlaQ), b.block, T);
}

// Prefill experts on the Flash-Next route: expert-sorted row compaction, the
// paired gate/up F16 GEMM with clamped SwiGLU, then the down GEMM, whose F16
// rows MoeCombineF16 sums with the shared expert.
void Executor::RoutedExpertsF16(const LayerWeights& w, int T) const {
  namespace qfn = gufo::models::qwen38_flash_next::rocm;
  const Scratch& b = *scratch_;
  const std::uint32_t slots = static_cast<std::uint32_t>(T) * kMoeUsed;
  ToHalf(b.y, b.x_half, static_cast<long long>(T) * kHidden, stream_);
  qfn::ExpertCounts(b.ids, b.counts, T, kMoeExperts, kMoeUsed, stream_);
  hipMemcpyAsync(b.host_counts, b.counts, kMoeExperts * sizeof(std::uint32_t),
                 hipMemcpyDeviceToHost, stream_);
  hipStreamSynchronize(stream_);
  const std::uint32_t pair_rows = kPairTileRows;
  const std::uint32_t down_rows = kDownTileRows;
  std::int32_t* host_pair = b.host_tiles;
  std::int32_t* host_down = b.host_tiles + b.tile_capacity;
  std::uint32_t pair_tiles = 0;
  std::uint32_t down_tiles = 0;
  for (int e = 0; e < kMoeExperts; ++e) {
    const std::uint32_t padded =
        (b.host_counts[e] + kPadRows - 1) / kPadRows * kPadRows;
    for (std::uint32_t j = 0; j < (padded + pair_rows - 1) / pair_rows; ++j) {
      host_pair[pair_tiles++] = static_cast<std::int32_t>(e | (j << 16));
    }
    for (std::uint32_t j = 0; j < (padded + down_rows - 1) / down_rows; ++j) {
      host_down[down_tiles++] = static_cast<std::int32_t>(e | (j << 16));
    }
  }
  hipMemcpyAsync(b.tiles_pair, host_pair, pair_tiles * sizeof(std::int32_t),
                 hipMemcpyHostToDevice, stream_);
  hipMemcpyAsync(b.tiles_down, host_down, down_tiles * sizeof(std::int32_t),
                 hipMemcpyHostToDevice, stream_);
  qfn::RoutedCompact(b.ids, b.counts, b.bounds, b.cursors, b.rows_token,
                     b.rows_slot, T, kMoeUsed, kMoeExperts, stream_);
  const auto weight_type = [](int code) {
    return code == kTypeIQ2_S ? qfn::WeightType::kIQ2_S
                              : qfn::WeightType::kIQ2_XXS;
  };
  qfn::RoutedGatedF16Gemm(
      w.exp_gate->data, w.exp_up->data, weight_type(w.exp_gate->type),
      static_cast<const __half*>(b.x_half), b.tiles_pair, pair_tiles, pair_rows,
      b.bounds, b.rows_token, b.rows_slot, static_cast<__half*>(b.act_half),
      kExpertFf, kHidden, stream_);
  qfn::RoutedF16Gemm(w.exp_down->data, weight_type(w.exp_down->type),
                     static_cast<const __half*>(b.act_half), b.tiles_down,
                     down_tiles, down_rows, b.bounds, b.rows_slot, b.rows_slot,
                     nullptr, nullptr, static_cast<__half*>(b.parts_half),
                     kHidden, kExpertFf, stream_);
  (void)slots;
}

void Executor::FfnBlock(std::uint32_t layer, int T) const {
  const LayerWeights& w = weights_[layer];
  const Scratch& b = *scratch_;
  arena_used_ = 0;
  const Act y = Prep(b.y, T, kHidden);
  if (w.dense) {
    Proj(*w.ffn_gate, y, b.gate, T);
    Proj(*w.ffn_up, y, b.up, T);
    SwigluClamp(b.gate, b.up, b.act, T * kDenseFf, stream_);
    Proj(*w.ffn_down, Prep(b.act, T, kDenseFf), b.ffn_out, T);
    return;
  }
  if (T >= kWideBatch && w.router->type == kTypeF32) {
    const auto handle = static_cast<hipblasHandle_t>(blas_);
    hipblasSetStream(handle, stream_);
    const float alpha = 1.0F;
    const float beta = 0.0F;
    hipblasSgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, kMoeExperts, T, kHidden,
                 &alpha, w.router->f32(), kHidden, b.y, kHidden, &beta,
                 b.rlogits, kMoeExperts);
  } else {
    Proj(*w.router, Act{b.y, nullptr}, b.rlogits, T);
  }
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
    RoutedExpertsF16(w, T);
  }
  Proj(*w.sh_gate, y, b.sg, T);
  Proj(*w.sh_up, y, b.su, T);
  SwigluClamp(b.sg, b.su, b.sact, T * kExpertFf, stream_);
  Proj(*w.sh_down, Prep(b.sact, T, kExpertFf), b.shared, T);
  if (T == 1) {
    MoeCombine(b.parts, b.wts, b.shared, b.ffn_out, stream_, T);
  } else {
    MoeCombineF16(b.parts_half, b.wts, b.shared, b.ffn_out, stream_, T);
  }
}

bool Executor::Forward(Session& session, const std::int32_t* tokens, int T,
                       int logit_rows, std::string* error) {
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

  if (T == 1) {
    const int pos = static_cast<int>(session.position);
    hipMemcpy(b.pos_dev, &pos, sizeof(int), hipMemcpyHostToDevice);
  }
  const auto run = [&]() {
    auto mark = [&](int bucket) { Mark(bucket); };
    profile_mark_ = std::chrono::steady_clock::now();
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
      const LayerWeights& w = weights_[layer];
      HcPre(b.x[cur], w.hc_attn_fn->f32(), w.hc_attn_scale->f32(),
            w.hc_attn_base->f32(), w.attn_norm->f32(), config_.hc_eps, b.y,
            b.post, b.comb, stream_, T);
      mark(0);
      if (w.kda) {
        KdaBlock(layer, session, T);
        mark(1);
      } else {
        DsaBlock(layer, session, T);
        mark(2);
      }
      HcPost(b.x[cur], b.block, b.post, b.comb, b.x[1 - cur], stream_, T);
      cur = 1 - cur;

      HcPre(b.x[cur], w.hc_ffn_fn->f32(), w.hc_ffn_scale->f32(),
            w.hc_ffn_base->f32(), w.ffn_norm->f32(), config_.hc_eps, b.y,
            b.post, b.comb, stream_, T);
      mark(0);
      FfnBlock(layer, T);
      mark(w.dense ? 3 : 4);
      HcPost(b.x[cur], b.ffn_out, b.post, b.comb, b.x[1 - cur], stream_, T);
      cur = 1 - cur;
      if (!dflash_taps_.empty()) {
        const auto tap =
            std::find(dflash_taps_.begin(), dflash_taps_.end(), layer);
        if (tap != dflash_taps_.end()) {
          const int rows = std::min<int>(T, options_.dflash_rows);
          MeanStreams(
              b.x[cur], T - rows, rows, b.dflash_capture,
              static_cast<long long>(dflash_taps_.size()) * kHidden,
              static_cast<long long>(tap - dflash_taps_.begin()) * kHidden,
              stream_);
          capture_rows_ = static_cast<std::uint32_t>(rows);
          capture_position_ =
              session.position + static_cast<std::uint32_t>(T - rows);
        }
      }
      mark(0);
    }
    last_residual_ =
        b.x[cur] + static_cast<std::size_t>(T - 1) * kStreams * kHidden;
    if (logit_rows > 0 && options_.head) {
      const DeviceTensor& norm = *Find("output_norm.weight");
      const DeviceTensor& out = *Find("output.weight");
      HeadNorm(b.x[cur] + static_cast<std::size_t>(T - logit_rows) * kStreams *
                              kHidden,
               norm.f32(), b.head_in, stream_, logit_rows);
      Proj(out, Act{b.head_in, nullptr}, b.logits, logit_rows);
      last_logit_rows_ = logit_rows;
    }
  };
  const bool graphable = T == 1 && !recording_ && !profile_;
  if (!graphable || !graph_warm_) {
    run();
    graph_warm_ = graph_warm_ || graphable;
  } else {
    if (graph_exec_ == nullptr || graph_session_ != session.id ||
        graph_logit_rows_ != logit_rows) {
      DropGraph();
      hipGraph_t graph = nullptr;
      if (capture_stream_ == nullptr) {
        hipStreamCreateWithFlags(&capture_stream_, hipStreamNonBlocking);
      }
      const hipStream_t launch_stream = stream_;
      stream_ = capture_stream_;
      hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal);
      run();
      const hipError_t ended = hipStreamEndCapture(stream_, &graph);
      stream_ = launch_stream;
      const hipError_t made =
          ended == hipSuccess
              ? hipGraphInstantiate(&graph_exec_, graph, nullptr, nullptr, 0)
              : ended;
      if (made != hipSuccess) {
        if (error != nullptr)
          *error = std::string("HIP graph capture failed: ") +
                   hipGetErrorString(made);
        return false;
      }
      hipGraphDestroy(graph);
      graph_session_ = session.id;
      graph_logit_rows_ = logit_rows;
    }
    hipGraphLaunch(graph_exec_, stream_);
    last_residual_ = b.x[0];
    if (logit_rows > 0 && options_.head) {
      last_logit_rows_ = logit_rows;
    }
    if (!dflash_taps_.empty()) {
      capture_rows_ = 1;
      capture_position_ = session.position;
    }
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

void Executor::DropGraph() {
  if (graph_exec_ != nullptr) {
    hipGraphExecDestroy(graph_exec_);
    graph_exec_ = nullptr;
  }
  graph_session_ = 0;
}

void Executor::Mark(int bucket) const {
  if (!profile_)
    return;
  hipStreamSynchronize(stream_);
  const auto now = std::chrono::steady_clock::now();
  profile_seconds_[bucket] +=
      std::chrono::duration<double>(now - profile_mark_).count();
  profile_mark_ = now;
}

void Executor::PrintProfile() {
  if (!profile_)
    return;
  static const char* kNames[] = {
      "hc",       "kda",        "dsa",          "ffn-dense",  "ffn-moe",
      "dsa:proj", "dsa:select", "dsa:headgemm", "dsa:attend", "dsa:out"};
  double total = 0.0;
  for (const double v : profile_seconds_)
    total += v;
  for (int i = 0; i < 10; ++i) {
    if (profile_seconds_[i] == 0.0)
      continue;
    std::printf("  %-10s %7.3f s %5.1f%%\n", kNames[i], profile_seconds_[i],
                100.0 * profile_seconds_[i] / std::max(total, 1e-9));
  }
  std::fill(std::begin(profile_seconds_), std::end(profile_seconds_), 0.0);
}

bool Executor::Evaluate(Session& session, std::int32_t token,
                        std::string* error) {
  return Forward(session, &token, 1, 1, error);
}

bool Executor::Prefill(Session& session, std::span<const std::int32_t> tokens,
                       std::string* error, std::size_t chunk_size) {
  if (tokens.empty()) {
    if (error != nullptr)
      *error = "empty prompt";
    return false;
  }
  const std::size_t chunk =
      chunk_size == 0
          ? options_.prefill_chunk
          : std::min<std::size_t>(chunk_size, options_.prefill_chunk);
  for (std::size_t at = 0; at < tokens.size(); at += chunk) {
    const std::size_t n = std::min(chunk, tokens.size() - at);
    const bool last = at + n == tokens.size();
    if (!Forward(session, tokens.data() + at, static_cast<int>(n), last ? 1 : 0,
                 error)) {
      return false;
    }
  }
  return true;
}

bool Executor::Verify(Session& session, std::span<const std::int32_t> tokens,
                      std::uint32_t rows, std::string* error) {
  if (rows == 0 || rows > options_.logit_rows || rows > tokens.size()) {
    if (error != nullptr)
      *error = "bad verification row count";
    return false;
  }
  return Forward(session, tokens.data(), static_cast<int>(tokens.size()),
                 static_cast<int>(rows), error);
}

std::vector<std::int32_t> Executor::ArgmaxRows(std::uint32_t rows) {
  std::vector<std::int32_t> out(rows);
  ::gufo::models::glm53_flash::rocm::ArgmaxRows(
      scratch_->logits, static_cast<int>(rows),
      static_cast<int>(config_.vocab_size), scratch_->argmax, stream_);
  hipMemcpyAsync(out.data(), scratch_->argmax, rows * sizeof(std::int32_t),
                 hipMemcpyDeviceToHost, stream_);
  hipStreamSynchronize(stream_);
  return out;
}

bool Executor::SaveState(Session& session, std::string* error) {
  if (session.saved_conv.empty()) {
    session.saved_conv.assign(layers_, nullptr);
    session.saved_state.assign(layers_, nullptr);
    for (std::uint32_t l = 0; l < layers_; ++l) {
      if (!weights_[l].kda)
        continue;
      if (!Alloc(&session.saved_conv[l], 3 * 3 * kKdaInner, error) ||
          !Alloc(
              &session.saved_state[l],
              static_cast<std::size_t>(kKdaHeads) * kKdaHeadDim * kKdaHeadDim,
              error)) {
        return false;
      }
    }
  }
  if (session.kda_log.empty()) {
    session.kda_log.resize(layers_);
    const std::size_t cap = options_.logit_rows;
    for (std::uint32_t l = 0; l < layers_; ++l) {
      if (!weights_[l].kda)
        continue;
      auto& log = session.kda_log[l];
      if (!Alloc(&log.raw_q, cap * kKdaInner, error) ||
          !Alloc(&log.raw_k, cap * kKdaInner, error) ||
          !Alloc(&log.raw_v, cap * kKdaInner, error) ||
          !Alloc(&log.decay, cap * kKdaInner, error) ||
          !Alloc(&log.beta, cap * kKdaHeads, error)) {
        return false;
      }
    }
  }
  session.saved_position = session.position;
  for (std::uint32_t l = 0; l < layers_; ++l) {
    if (!weights_[l].kda)
      continue;
    hipMemcpyAsync(session.saved_conv[l], session.layers[l].conv,
                   3 * 3 * kKdaInner * sizeof(float), hipMemcpyDeviceToDevice,
                   stream_);
    hipMemcpyAsync(session.saved_state[l], session.layers[l].state,
                   static_cast<std::size_t>(kKdaHeads) * kKdaHeadDim *
                       kKdaHeadDim * sizeof(float),
                   hipMemcpyDeviceToDevice, stream_);
  }
  return true;
}

bool Executor::Rollback(Session& session, std::uint32_t keep,
                        std::string* error) {
  if (session.kda_log.empty() || keep == 0 || keep > options_.logit_rows) {
    if (error != nullptr)
      *error = "nothing recorded to roll back to";
    return false;
  }
  RestoreState(session);
  Scratch& b = *scratch_;
  for (std::uint32_t l = 0; l < layers_; ++l) {
    if (!weights_[l].kda)
      continue;
    const LayerWeights& w = weights_[l];
    const auto& log = session.kda_log[l];
    const std::size_t bytes =
        static_cast<std::size_t>(keep) * kKdaInner * sizeof(float);
    hipMemcpyAsync(b.kq, log.raw_q, bytes, hipMemcpyDeviceToDevice, stream_);
    hipMemcpyAsync(b.kk, log.raw_k, bytes, hipMemcpyDeviceToDevice, stream_);
    hipMemcpyAsync(b.kv, log.raw_v, bytes, hipMemcpyDeviceToDevice, stream_);
    KdaConv(b.kq, b.kk, b.kv, session.layers[l].conv, w.conv_q->f32(),
            w.conv_k->f32(), w.conv_v->f32(), stream_, static_cast<int>(keep));
    KdaRecur(b.kq, b.kk, b.kv, log.decay, log.beta, session.layers[l].state,
             b.o, stream_, static_cast<int>(keep));
  }
  session.position = session.saved_position + keep;
  if (hipStreamSynchronize(stream_) != hipSuccess) {
    if (error != nullptr)
      *error = "HIP error during rollback";
    return false;
  }
  return true;
}

void Executor::RestoreState(Session& session) {
  for (std::uint32_t l = 0; l < layers_; ++l) {
    if (!weights_[l].kda || session.saved_conv.empty())
      continue;
    hipMemcpyAsync(session.layers[l].conv, session.saved_conv[l],
                   3 * 3 * kKdaInner * sizeof(float), hipMemcpyDeviceToDevice,
                   stream_);
    hipMemcpyAsync(session.layers[l].state, session.saved_state[l],
                   static_cast<std::size_t>(kKdaHeads) * kKdaHeadDim *
                       kKdaHeadDim * sizeof(float),
                   hipMemcpyDeviceToDevice, stream_);
  }
  hipStreamSynchronize(stream_);
}

const float* Executor::DflashCapture() const {
  return scratch_->dflash_capture;
}

const DeviceTensor* Executor::TokenEmbedding() const {
  return token_embd_dev_.data != nullptr ? &token_embd_dev_ : nullptr;
}

const DeviceTensor* Executor::Output() const {
  return Find("output.weight");
}

std::vector<float> Executor::Logits(const Session&) {
  std::vector<float> out(config_.vocab_size);
  hipMemcpy(out.data(),
            scratch_->logits + static_cast<std::size_t>(last_logit_rows_ - 1) *
                                   config_.vocab_size,
            out.size() * sizeof(float), hipMemcpyDeviceToHost);
  return out;
}

std::vector<float> Executor::Residual(const Session&) {
  std::vector<float> out(kStreams * kHidden);
  hipMemcpy(out.data(), last_residual_, out.size() * sizeof(float),
            hipMemcpyDeviceToHost);
  return out;
}

}  // namespace gufo::models::glm53_flash::rocm
