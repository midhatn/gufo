#ifndef GUFO_MODELS_FLINT_GPU_ENGINE_HPP_
#define GUFO_MODELS_FLINT_GPU_ENGINE_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "src/models/flint/runtime/config.hpp"

namespace gufo::models::flint::gpu {

struct EngineOptions {
  /// Tokens one prefill pass processes.
  std::uint32_t prefill_tile{4096};
};

class Model;
struct ModelState;

/// One conversation: KV cache, position and the logits of the last token.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Appends `tokens` and leaves the logits of the last one in Logits().
  [[nodiscard]] bool Append(std::span<const std::int32_t> tokens,
                            std::string* error_msg);

  /// Appends 1..kMaxVerifyRows tokens and keeps the logits of every row:
  /// row i is the distribution after tokens[i]. Used to verify drafts.
  [[nodiscard]] bool AppendAll(std::span<const std::int32_t> tokens,
                               std::string* error_msg);
  static constexpr std::uint32_t kMaxVerifyRows = 5;

  /// Drops cache rows past `position`; later appends overwrite them.
  void Rewind(std::uint32_t position) noexcept;

  [[nodiscard]] std::span<const float> Logits() const noexcept;
  /// Row `row` of the last AppendAll (or Append of up to kMaxVerifyRows).
  [[nodiscard]] std::span<const float> Logits(std::uint32_t row) const noexcept;
  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::uint32_t Capacity() const noexcept;

  /// Forgets everything: position 0, empty cache.
  void Reset() noexcept;

  /// Device bytes of the KV caches and key summaries (scratch excluded).
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;

  /// Exact continuation at the current position: the K and V rows of every
  /// layer plus the last logits, so a restored state can sample at once.
  /// Bumped when the layout changes.
  static constexpr std::uint32_t kSnapshotPayloadVersion = 1;
  [[nodiscard]] std::size_t SnapshotBytes() const noexcept;
  [[nodiscard]] bool SaveSnapshot(std::span<std::uint8_t> out,
                                  std::string* error_msg) const;
  /// Replaces the session state with a payload from SaveSnapshot (of this
  /// model). The position becomes the payload's.
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::string* error_msg);

private:
  friend class Model;
  Session();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// The Flint weights resident on the GPU, shared by sessions.
class Model final {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const EngineOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr);

  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// Device bytes of one session with `max_context` positions.
  [[nodiscard]] std::size_t SessionBytes(
      std::uint32_t max_context) const noexcept;
  /// Bytes of a snapshot at `tokens` positions.
  [[nodiscard]] std::size_t SnapshotBytes(std::uint32_t tokens) const noexcept;

private:
  Model() = default;
  std::shared_ptr<ModelState> state_;
};

}  // namespace gufo::models::flint::gpu

#endif  // GUFO_MODELS_FLINT_GPU_ENGINE_HPP_
