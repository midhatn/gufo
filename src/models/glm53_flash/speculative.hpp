#ifndef GUFO_MODELS_GLM53_FLASH_SPECULATIVE_HPP_
#define GUFO_MODELS_GLM53_FLASH_SPECULATIVE_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/kernels/rocm/executor.hpp"

namespace gufo::hip {
class QwenDFlashGpuExecutor;
class QwenDFlashGpuModel;
}  // namespace gufo::hip

namespace gufo::models::glm53_flash {

// Greedy speculative decoding of GLM-5.3-Flash with the DFlash-2 block
// diffusion drafter. The drafter reads the stream-mean residual after the
// trunk layers it names; each step drafts a block, verifies it with one
// batched target forward, and replays the accepted prefix after a rejection.
class Speculator {
public:
  struct Stats {
    std::uint64_t steps{0};
    std::uint64_t drafted{0};
    std::uint64_t accepted{0};
    std::uint64_t emitted{0};
    std::uint64_t replays{0};
    std::uint64_t accepted_histogram[16]{};  // steps by accepted draft count
  };

  // Trunk taps and block size the drafter file asks for; the executor must be
  // created with these before Create().
  struct DrafterInfo {
    std::vector<std::uint32_t> target_layers;
    std::uint32_t block_size{0};
    std::uint32_t sliding_window{0};
  };
  static bool ReadDrafterInfo(const std::string& path, DrafterInfo* info,
                              std::string* error);

  static std::unique_ptr<Speculator> Create(rocm::Executor& executor,
                                            const std::string& dflash_path,
                                            std::uint32_t max_context,
                                            std::string* error);
  ~Speculator();

  // Generates up to max_new tokens after `prompt`, stopping at a token in
  // `stop`. Each emitted token goes to on_token (return false to stop).
  bool Generate(rocm::Executor::Session& session,
                std::span<const std::int32_t> prompt, std::size_t max_new,
                std::span<const std::int32_t> stop,
                const std::function<bool(std::int32_t)>& on_token,
                std::string* error);

  [[nodiscard]] const Stats& stats() const { return stats_; }
  [[nodiscard]] std::uint32_t block_size() const { return info_.block_size; }

private:
  Speculator() = default;

  bool InjectCapture(std::string* error);
  bool InjectCaptureRows(std::uint32_t rows, std::string* error);

  rocm::Executor* executor_{nullptr};
  DrafterInfo info_;
  std::shared_ptr<const hip::QwenDFlashGpuModel> model_;
  std::unique_ptr<hip::QwenDFlashGpuExecutor> drafter_;
  std::uint32_t max_context_{0};
  Stats stats_;
};

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_SPECULATIVE_HPP_
