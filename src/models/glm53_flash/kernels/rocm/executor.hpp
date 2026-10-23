#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/kernels/rocm/dsa.hpp"
#include "src/models/glm53_flash/runtime/config.hpp"

namespace gufo::models::glm53_flash::rocm {

struct ExecutorOptions {
  // Trunk layers to run (0 = all). A prefix is useful for layer-level checks.
  std::uint32_t layers{0};
  std::uint32_t max_context{4096};
  std::uint32_t top_pools{kDsaTopPools};
  // Skip the output head (embedding to residual only).
  bool head{true};
};

struct DeviceTensor {
  void* data{nullptr};
  int type{-1};
  std::uint32_t k{0};
  std::uint32_t rows{0};
  std::uint32_t experts{1};
  [[nodiscard]] const float* f32() const {
    return static_cast<const float*>(data);
  }
};

// GPU decode engine for GLM-5.3-Flash: one token per step on one stream.
class Executor {
public:
  struct LayerState {
    float* conv{nullptr};   // KDA: [3][3][8192]
    float* state{nullptr};  // KDA: [64][128][128]
    DsaCache cache{};       // DSA
  };
  struct Session {
    float* residual[2]{nullptr, nullptr};  // [4][4096], ping-pong
    int current{0};
    std::uint32_t position{0};
    std::vector<LayerState> layers;
    ~Session();
  };

  static std::unique_ptr<Executor> Create(const std::string& path,
                                          const ExecutorOptions& options,
                                          std::string* error);
  ~Executor();

  std::unique_ptr<Session> CreateSession(std::string* error);

  // Runs one token through the trunk (and head unless disabled).
  bool Evaluate(Session& session, std::int32_t token, std::string* error);

  // Host copies; valid after Evaluate returns.
  std::vector<float> Logits(const Session& session);
  std::vector<float> Residual(const Session& session);

  [[nodiscard]] std::uint32_t layers() const { return layers_; }

private:
  Executor() = default;

  struct LayerWeights;
  struct Scratch;

  bool CopyToDevice(void* dst, const core::GgufTensorInfo& t,
                    std::string* error);
  bool Upload(const std::string& name, std::string* error);
  const DeviceTensor* Find(const std::string& name) const;
  bool Resolve(std::string* error);
  bool AllocScratch(std::string* error);

  void Matvec(const DeviceTensor& w, const float* x, float* out) const;
  void KdaBlock(std::uint32_t layer, Session& s, const float* y,
                float* block) const;
  void DsaBlock(std::uint32_t layer, Session& s, const float* y,
                float* block) const;
  void FfnBlock(std::uint32_t layer, const float* y, float* out) const;

  std::unique_ptr<core::GgufReader> reader_;
  Config config_;
  ExecutorOptions options_;
  std::uint32_t layers_{0};
  std::unordered_map<std::string, DeviceTensor> tensors_;
  std::vector<LayerWeights> weights_;
  std::unique_ptr<Scratch> scratch_;
  hipStream_t stream_{nullptr};
  void* staging_{nullptr};
};

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_EXECUTOR_HPP_
