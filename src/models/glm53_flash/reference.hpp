#ifndef GUFO_MODELS_GLM53_FLASH_REFERENCE_HPP_
#define GUFO_MODELS_GLM53_FLASH_REFERENCE_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/runtime/config.hpp"

namespace gufo::models::glm53_flash {

// Correctness-first float32 forward for GLM-5.3-Flash on CPU. One row at a
// time, every format dequantized on demand; mirrors the cross-validated
// reference (tests/models/glm53_flash). Not tuned; GPU paths replace it.
class ReferenceModel {
public:
  static std::unique_ptr<ReferenceModel> Open(const std::string& path,
                                              std::string* error);

  struct LayerState {
    std::vector<float> conv;    // 3 * 3 * d_inner (KDA conv history)
    std::vector<float> state;   // heads*hd*hd (KDA delta state)
    std::vector<float> lat;     // n * 512 (MLA latent)
    std::vector<float> ik;      // n * 128 (indexer key, normalized)
    std::vector<float> gate;    // n * 128 (compressor gate)
    std::vector<float> pooled;  // npool * 128
  };

  // Streaming state for one sequence. Evaluate() advances it one token.
  struct Session {
    std::vector<float> x;       // hc*h residual streams
    std::vector<float> logits;  // vocab
    std::uint32_t position{0};
    std::vector<LayerState> layers;
  };

  std::unique_ptr<Session> CreateSession() const;

  // Runs one token through all layers and returns fresh logits.
  const std::vector<float>& Evaluate(Session& session,
                                     std::int32_t token) const;

  std::uint32_t n_layers() const;

private:
  ReferenceModel() = default;

  // F32 tensor materialized on demand (or row-view only for huge tensors).
  struct Loaded {
  public:
    Loaded() = default;
    explicit Loaded(const gufo::core::GgufTensorInfo* t);
    static Loaded RowView(const gufo::core::GgufTensorInfo* t);

    void DequantizeRow(std::uint64_t row, float* out) const;
    void MatVec(std::span<const float> x, std::span<float> y) const;
    void MatVecRows(std::uint64_t row0, std::uint64_t nrows,
                    std::span<const float> x, std::span<float> y) const;
    void MatVecView(std::uint64_t row0, std::uint64_t nrows,
                    std::span<const float> x, std::span<float> y) const;

    std::uint64_t rows() const { return rows_; }
    const std::vector<float>& data() const { return data_; }
    const float* w() const { return data_.data(); }

  private:
    void Setup(const gufo::core::GgufTensorInfo* t);

    const gufo::core::GgufTensorInfo* t_{nullptr};
    std::uint64_t rows_{0}, cols_{0};
    std::vector<float> data_;
  };

  const Loaded& Load(const std::string& name) const;
  const Loaded& LoadRow(const std::string& name) const;

  void HcMix(std::span<const float> x, const std::string& prefix, float eps,
             std::vector<float>& pre, std::vector<float>& post,
             std::vector<float>& comb) const;
  void AttnBlock(std::uint32_t layer, Session& session, bool dsa,
                 std::span<const float> xn, std::span<float> block) const;
  void FfnBlock(std::uint32_t layer, std::span<const float> ffn_in,
                std::span<float> ffn_out) const;

  std::unique_ptr<gufo::core::GgufReader> reader_;
  std::unordered_map<std::string, const gufo::core::GgufTensorInfo*> tensors_;
  Config config_;
  mutable std::unordered_map<std::string, Loaded> cache_;
};

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_REFERENCE_HPP_
