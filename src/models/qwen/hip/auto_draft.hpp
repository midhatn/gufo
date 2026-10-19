#ifndef GUFO_MODELS_QWEN_HIP_AUTO_DRAFT_HPP_
#define GUFO_MODELS_QWEN_HIP_AUTO_DRAFT_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/mtp.hpp"

#if defined(ENGINE_ENABLE_HIP)
namespace gufo::hip {

/// Runs a DFlash2 draft and an MTP head side by side and proposes with
/// whichever has been committing more tokens per second in the current request.
/// Both stay in sync with every committed token, so the choice can change at
/// any step. Sampled requests always use DFlash2, the only draft that reports
/// its proposal distribution.
class QwenAutoDraftBackend final : public speculative::IDraftBackend {
public:
  [[nodiscard]] static std::unique_ptr<QwenAutoDraftBackend> Create(
      std::unique_ptr<QwenDFlashGpuDraftBackend> dflash,
      std::unique_ptr<QwenMtpGpuDraftBackend> mtp, std::uint32_t num_layers,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "QwenAutoDraftBackend";
  }
  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }
  [[nodiscard]] std::span<const std::uint32_t> TargetHiddenLayerIds()
      const noexcept override {
    return layer_ids_;
  }
  [[nodiscard]] bool SupportsSampledProposals() const noexcept override {
    return dflash_->SupportsSampledProposals();
  }

  [[nodiscard]] bool PrimeTargetContext(
      const speculative::DraftTargetContext& context) override;
  [[nodiscard]] bool AppendTargetContext(
      const speculative::DraftTargetContext& context,
      std::uint32_t position) override;
  [[nodiscard]] speculative::DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;
  [[nodiscard]] speculative::DraftProposal ProposeSampled(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
      std::uint64_t* rng_state) override;
  void UpdateTargetHidden(std::span<const float> hidden) override;
  void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                      tokenization::TokenId correction_token) override;
  void DiscardPendingTargetContext(std::uint32_t position) override;
  [[nodiscard]] std::pair<std::size_t, std::size_t> RequestMemoryBytes()
      const override;
  [[nodiscard]] std::size_t SnapshotPayloadBytes() const override;
  [[nodiscard]] std::unique_ptr<speculative::IDraftBackendSnapshot> Snapshot()
      const override;
  void RestoreSnapshot(
      const speculative::IDraftBackendSnapshot& snapshot) override;
  void RestorePersistentSnapshot(
      std::span<const std::uint8_t> payload) override;
  void Reset() noexcept override;
  void BeginRequest() noexcept override;

private:
  enum class Which : std::uint8_t { kDFlash, kMtp };

  struct Rate {
    double tokens_per_second{0.0};
    std::uint32_t samples{0};
    std::uint32_t idle_steps{0};
  };

  QwenAutoDraftBackend(std::unique_ptr<QwenDFlashGpuDraftBackend> dflash,
                       std::unique_ptr<QwenMtpGpuDraftBackend> mtp,
                       std::vector<std::uint32_t> layer_ids,
                       std::vector<std::size_t> dflash_columns,
                       std::size_t mtp_column, std::size_t hidden_size);

  [[nodiscard]] Which Choose();
  void RecordStep(Which which, std::size_t committed_tokens);
  std::size_t pending_committed_{0};
  [[nodiscard]] std::vector<float> Columns(
      std::span<const float> rows, std::size_t row_count,
      std::span<const std::size_t> columns) const;

  std::unique_ptr<QwenDFlashGpuDraftBackend> dflash_;
  std::unique_ptr<QwenMtpGpuDraftBackend> mtp_;
  std::vector<std::uint32_t> layer_ids_;
  std::vector<std::size_t> dflash_columns_;
  std::size_t mtp_column_;
  std::size_t hidden_size_;

  Rate rates_[2];
  Which proposer_{Which::kMtp};
  bool proposal_open_{false};
  tokenization::TokenId step_frontier_{0};
  std::uint32_t step_position_{0};
  std::vector<float> mtp_rows_;
  std::chrono::steady_clock::time_point last_propose_{};
  Which last_which_{Which::kMtp};
  bool have_last_{false};
  std::uint32_t steps_in_request_{0};
  std::uint32_t exploring_{0};
};

}  // namespace gufo::hip
#endif
#endif  // GUFO_MODELS_QWEN_HIP_AUTO_DRAFT_HPP_
