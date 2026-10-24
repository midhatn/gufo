#include "src/models/glm53_flash/speculative.hpp"

#include <algorithm>
#include <cmath>

#include "src/models/qwen/dflash_weights.hpp"
#include "src/models/qwen/hip/dflash.hpp"

namespace gufo::models::glm53_flash {

namespace {

models::QwenTensorRef Ref(const rocm::DeviceTensor& t) {
  models::QwenTensorRef ref;
  ref.data = t.data;
  ref.type = core::GgmlType::kQ6_K;
  ref.num_elements = static_cast<std::size_t>(t.k) * t.rows;
  return ref;
}

}  // namespace

bool Speculator::ReadDrafterInfo(const std::string& path, DrafterInfo* info,
                                 std::string* error) {
  auto reader = core::GgufReader::OpenFile(path, error);
  if (!reader) {
    return false;
  }
  auto weights = speculative::QwenDFlashWeights::LoadFromGguf(*reader, error);
  if (!weights) {
    return false;
  }
  info->target_layers = weights->dflash_config.target_layer_ids;
  info->block_size = weights->dflash_config.block_size;
  info->sliding_window = weights->dflash_config.sliding_window;
  return true;
}

std::unique_ptr<Speculator> Speculator::Create(rocm::Executor& executor,
                                               const std::string& dflash_path,
                                               std::uint32_t max_context,
                                               std::string* error) {
  std::unique_ptr<Speculator> s(new Speculator());
  s->executor_ = &executor;
  s->max_context_ = max_context;
  if (!ReadDrafterInfo(dflash_path, &s->info_, error)) {
    return nullptr;
  }
  const auto* embedding = executor.TokenEmbedding();
  const auto* output = executor.Output();
  if (embedding == nullptr || output == nullptr) {
    if (error != nullptr)
      *error = "executor was not created with drafter taps";
    return nullptr;
  }
  std::shared_ptr<const core::GgufReader> reader =
      core::GgufReader::OpenFile(dflash_path, error);
  if (!reader) {
    return nullptr;
  }
  const auto& config = executor.config();
  s->model_ = hip::QwenDFlashGpuModel::Create(
      std::move(reader), config.hidden_size, config.vocab_size,
      config.TrunkLayers(), Ref(*embedding), Ref(*output), error);
  if (!s->model_) {
    return nullptr;
  }
  s->drafter_ =
      hip::QwenDFlashGpuExecutor::Create(s->model_, max_context, error);
  if (!s->drafter_) {
    return nullptr;
  }
  return s;
}

Speculator::~Speculator() = default;

bool Speculator::InjectCaptureRows(std::uint32_t rows, std::string* error) {
  if (executor_->DflashRows() < rows) {
    if (error != nullptr)
      *error = "drafter capture is shorter than the committed rows";
    return false;
  }
  if (!drafter_->InjectTargetContextDevice(executor_->DflashCapture(),
                                           executor_->DflashPosition(), rows)) {
    if (error != nullptr)
      *error = "drafter context injection failed";
    return false;
  }
  return true;
}

bool Speculator::InjectCapture(std::string* error) {
  if (executor_->DflashRows() == 0) {
    return true;
  }
  if (!drafter_->InjectTargetContextDevice(executor_->DflashCapture(),
                                           executor_->DflashPosition(),
                                           executor_->DflashRows())) {
    if (error != nullptr)
      *error = "drafter context injection failed";
    return false;
  }
  return true;
}

namespace {

constexpr std::uint32_t kNgramOrder = 3;
constexpr std::uint32_t kNgramMinDraft = 2;
constexpr float kPlainBelow = 0.8F;
constexpr std::uint32_t kProbeEvery = 6;

// Tokens that followed the latest earlier occurrence of the history's last
// kNgramOrder tokens, at most `limit` of them.
std::vector<std::int32_t> LookupNgram(std::span<const std::int32_t> history,
                                      std::uint32_t limit) {
  const std::size_t n = history.size();
  if (n <= kNgramOrder) {
    return {};
  }
  const auto* tail = history.data() + n - kNgramOrder;
  for (std::size_t at = n - kNgramOrder; at-- > 0;) {
    if (!std::equal(tail, tail + kNgramOrder, history.data() + at)) {
      continue;
    }
    const std::size_t first = at + kNgramOrder;
    const std::size_t count = std::min<std::size_t>(limit, n - first);
    return {history.begin() + first, history.begin() + first + count};
  }
  return {};
}

}  // namespace

bool Speculator::Generate(rocm::Executor::Session& session,
                          std::span<const std::int32_t> prompt,
                          std::size_t max_new,
                          std::span<const std::int32_t> stop,
                          const std::function<bool(std::int32_t)>& on_token,
                          std::string* error) {
  if (prompt.empty()) {
    if (error != nullptr)
      *error = "empty prompt";
    return false;
  }
  drafter_->Reset();
  const auto is_stop = [&](std::int32_t t) {
    return std::find(stop.begin(), stop.end(), t) != stop.end();
  };

  // Prefill chunk by chunk so every chunk's taps reach the drafter.
  const std::size_t chunk = executor_->PrefillChunk();
  for (std::size_t at = 0; at < prompt.size(); at += chunk) {
    const std::size_t n = std::min(chunk, prompt.size() - at);
    const bool last = at + n == prompt.size();
    if (!executor_->Step(session, prompt.subspan(at, n), last ? 1 : 0, error) ||
        !InjectCapture(error)) {
      return false;
    }
  }
  std::int32_t anchor = executor_->ArgmaxRows(1)[0];
  std::size_t emitted = 0;
  std::vector<std::int32_t> history(prompt.begin(), prompt.end());
  const auto emit = [&](std::int32_t token) {
    ++emitted;
    ++stats_.emitted;
    history.push_back(token);
    return on_token(token) && !is_stop(token);
  };
  if (!emit(anchor) || emitted >= max_new) {
    return true;
  }

  const std::uint32_t max_draft = info_.block_size - 1;
  // Verification cost grows with every token (more distinct experts), while
  // chat-like text rarely accepts more than a few drafts: track the recent
  // accepted count and verify only a little beyond it.
  float accepted_ema = static_cast<float>(max_draft);
  std::uint32_t plain_steps = 0;
  while (emitted < max_new) {
    const std::uint32_t position = session.position;  // the anchor's position
    if (position + max_draft + 1 >= max_context_) {
      break;
    }
    std::vector<std::int32_t> chain{anchor};
    bool drafted_by_model = false;
    const auto lookup = LookupNgram(history, max_draft);
    if (lookup.size() >= kNgramMinDraft) {
      chain.insert(chain.end(), lookup.begin(), lookup.end());
    } else if (accepted_ema < kPlainBelow && ++plain_steps % kProbeEvery != 0) {
      // The drafter has been missing: a plain step costs less than a
      // verification block that is mostly rejected.
    } else {
      const auto draft = drafter_->ForwardBlock(
          static_cast<tokenization::TokenId>(anchor), position, max_draft);
      if (draft.empty()) {
        if (error != nullptr)
          *error = "drafter returned no tokens";
        return false;
      }
      const std::uint32_t wanted = std::clamp<std::uint32_t>(
          static_cast<std::uint32_t>(std::ceil(accepted_ema)) + 1, 2,
          max_draft);
      const std::uint32_t take = std::min<std::uint32_t>(
          wanted, static_cast<std::uint32_t>(draft.size()));
      for (std::uint32_t i = 0; i < take; ++i) {
        chain.push_back(static_cast<std::int32_t>(draft[i]));
      }
      drafted_by_model = true;
    }
    const std::uint32_t k = static_cast<std::uint32_t>(chain.size()) - 1;

    executor_->SetRecording(k > 0);
    const bool ok = (k == 0 || executor_->SaveState(session, error)) &&
                    executor_->Step(session, chain, k + 1, error);
    executor_->SetRecording(false);
    if (!ok) {
      return false;
    }
    const auto target = executor_->ArgmaxRows(k + 1);
    std::uint32_t accepted = 0;
    while (accepted < k && chain[accepted + 1] == target[accepted]) {
      ++accepted;
    }
    ++stats_.steps;
    stats_.drafted += k;
    stats_.accepted += accepted;
    if (drafted_by_model) {
      accepted_ema = 0.7F * accepted_ema + 0.3F * static_cast<float>(accepted);
    }
    ++stats_.accepted_histogram[std::min<std::uint32_t>(accepted, 15)];

    if (accepted < k) {
      // The batch consumed rejected tokens: rebuild the recurrent state from
      // the recorded operands for the committed prefix (anchor + accepted
      // drafts). The rows' drafter features are already in the capture.
      if (!executor_->Rollback(session, accepted + 1, error)) {
        return false;
      }
      ++stats_.replays;
    }
    if (!InjectCaptureRows(accepted + 1, error)) {
      return false;
    }

    bool stopped = false;
    for (std::uint32_t i = 0; i < accepted && emitted < max_new; ++i) {
      if (!emit(chain[i + 1])) {
        stopped = true;
        break;
      }
    }
    if (stopped || emitted >= max_new) {
      break;
    }
    anchor = target[accepted];
    if (!emit(anchor)) {
      break;
    }
  }
  return true;
}

}  // namespace gufo::models::glm53_flash
