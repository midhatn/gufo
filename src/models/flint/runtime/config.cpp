#include "src/models/flint/runtime/config.hpp"

#include <cmath>
#include <string_view>
#include <utility>

namespace gufo::models::flint {
namespace {

constexpr std::string_view kArchitecture = "flint";

struct Reader {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  std::uint32_t U32(std::string_view key) {
    const auto value = gguf.GetMetadataUint32(key);
    if (!value.has_value()) {
      Fail("missing or invalid GGUF integer " + std::string(key));
      return 0;
    }
    return *value;
  }

  float F32(std::string_view key) {
    const auto value = gguf.GetMetadataFloat32(key);
    if (!value || !std::isfinite(*value)) {
      Fail("missing or invalid GGUF float " + std::string(key));
      return 0.0F;
    }
    return *value;
  }

  bool Bool(std::string_view key, bool fallback) {
    const auto value = gguf.GetMetadataBool(key);
    if (!value.has_value()) {
      Fail("missing or invalid GGUF bool " + std::string(key));
      return fallback;
    }
    return *value;
  }

  std::vector<bool> BoolArray(std::string_view key) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<bool> out;
    if (meta == nullptr) {
      Fail("missing GGUF array " + std::string(key));
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out.reserve(u->size());
      for (const auto value : *u) {
        out.push_back(value != 0);
      }
    } else if (const auto* b =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      out.reserve(b->size());
      for (const auto value : *b) {
        out.push_back(value != 0);
      }
    } else {
      Fail("GGUF array must contain integers: " + std::string(key));
    }
    return out;
  }
};

void Expect(Reader& r, std::string_view name, std::uint64_t value,
            std::uint64_t expected) {
  if (value != expected) {
    r.Fail("Flint: " + std::string(name) + " is " + std::to_string(value) +
           ", this build supports " + std::to_string(expected));
  }
}

}  // namespace

std::optional<Config> ParseConfig(const core::GgufReader& reader,
                                  std::string* error_msg) {
  Reader r{reader, error_msg};
  const auto architecture = reader.GetMetadataString("general.architecture");
  if (!architecture || *architecture != kArchitecture) {
    r.Fail("artifact is not flint");
    return std::nullopt;
  }

  Config c;
  const auto key = [](std::string_view name) {
    return std::string(kArchitecture) + "." + std::string(name);
  };
  c.num_layers = r.U32(key("block_count"));
  c.context_length = r.U32(key("context_length"));
  c.hidden_size = r.U32(key("embedding_length"));
  c.dense_ff = r.U32(key("feed_forward_length"));
  c.num_heads = r.U32(key("attention.head_count"));
  c.num_kv = r.U32(key("attention.head_count_kv"));
  c.head_dim = r.U32(key("attention.key_length"));
  c.expert_count = r.U32(key("expert_count"));
  c.expert_used = r.U32(key("expert_used_count"));
  c.expert_ff = r.U32(key("expert_feed_forward_length"));
  c.shared_ff = r.U32(key("expert_shared_feed_forward_length"));
  c.leading_dense = r.U32(key("leading_dense_block_count"));
  c.expert_shared = r.U32(key("expert_shared_count"));
  c.sliding_window = r.U32(key("attention.sliding_window"));
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"));
  c.rope_base = r.F32(key("rope.freq_base"));
  c.expert_weights_scale = r.F32(key("expert_weights_scale"));
  c.expert_weights_norm = r.Bool(key("expert_weights_norm"), true);
  c.expert_gating_func = r.U32(key("expert_gating_func"));
  c.is_swa = r.BoolArray(key("attention.sliding_window_pattern"));
  c.rope_pattern = r.BoolArray(key("attention.rope_pattern"));

  if (const auto value_length =
          reader.GetMetadataUint32(key("attention.value_length"))) {
    if (*value_length != c.head_dim) {
      r.Fail("Flint: attention.value_length differs from key_length");
    }
  } else {
    r.Fail("missing or invalid GGUF integer " + key("attention.value_length"));
  }
  {
    const auto tokens = reader.GetMetadataStringArray("tokenizer.ggml.tokens");
    if (tokens.empty()) {
      r.Fail("missing tokenizer.ggml.tokens");
    } else {
      c.vocab_size = static_cast<std::uint32_t>(tokens.size());
    }
  }
  if (!r.ok) {
    return std::nullopt;
  }

  Expect(r, "block_count", c.num_layers, 49);
  Expect(r, "embedding_length", c.hidden_size, 2048);
  Expect(r, "feed_forward_length", c.dense_ff, 6144);
  Expect(r, "attention.head_count", c.num_heads, 32);
  Expect(r, "attention.head_count_kv", c.num_kv, 4);
  Expect(r, "attention.key_length", c.head_dim, 128);
  Expect(r, "expert_count", c.expert_count, 256);
  Expect(r, "expert_used_count", c.expert_used, 6);
  Expect(r, "expert_feed_forward_length", c.expert_ff, 512);
  Expect(r, "expert_shared_feed_forward_length", c.shared_ff, 1024);
  Expect(r, "leading_dense_block_count", c.leading_dense, 2);
  Expect(r, "expert_shared_count", c.expert_shared, 2);
  Expect(r, "attention.sliding_window", c.sliding_window, 513);
  Expect(r, "vocab_size", c.vocab_size, 150144);
  Expect(r, "expert_gating_func", c.expert_gating_func, 2);
  Expect(r, "sliding_window_pattern entries", c.is_swa.size(), c.num_layers);
  Expect(r, "rope_pattern entries", c.rope_pattern.size(), c.num_layers);
  if (!r.ok) {
    return std::nullopt;
  }
  return c;
}

}  // namespace gufo::models::flint
