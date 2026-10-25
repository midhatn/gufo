#ifndef GUFO_MODELS_FLINT_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_FLINT_CHAT_TEMPLATE_HPP_

#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::flint {

// Flint control tokens as written in the artifact's chat template.
inline constexpr std::string_view kSystemTag = "<|system|>";
inline constexpr std::string_view kUserTag = "<|user|>";
inline constexpr std::string_view kAssistantTag = "<|assistant|>";
inline constexpr std::string_view kThinkOpen = "<think>";
inline constexpr std::string_view kThinkClose = "</think>";

struct ChatMessage {
  std::string role;  // system | developer | user | assistant | tool
  std::string content;
  std::string reasoning_content;
};

struct ChatTemplateOptions {
  bool enable_thinking{false};
};

// Mirrors the GGUF Jinja template for system/user/assistant turns with
// add_generation_prompt. Tool preamble is omitted (Phase 1 prompt path).
[[nodiscard]] std::string RenderPrompt(const std::vector<ChatMessage>& messages,
                                       const ChatTemplateOptions& options = {});

// Convenience: system + single user turn.
[[nodiscard]] std::string RenderPrompt(std::string_view system,
                                       std::string_view user,
                                       const ChatTemplateOptions& options = {});

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_CHAT_TEMPLATE_HPP_
