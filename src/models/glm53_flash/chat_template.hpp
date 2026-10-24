#ifndef GUFO_MODELS_GLM53_FLASH_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_GLM53_FLASH_CHAT_TEMPLATE_HPP_

#include <span>
#include <string>
#include <string_view>

#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::models::glm53_flash {

// GLM-5.3 control tokens as written in the artifact's chat template.
inline constexpr std::string_view kSystemTag = "<|system|>";
inline constexpr std::string_view kUserTag = "<|user|>";
inline constexpr std::string_view kAssistantTag = "<|assistant|>";
inline constexpr std::string_view kThinkOpen = "<think>";

// Mirrors the GGUF Jinja template for a system/user turn with
// add_generation_prompt. Only "low" and "high" are honored; anything else
// renders as "Max", like the template.
[[nodiscard]] std::string RenderPrompt(std::string_view system,
                                       std::string_view user,
                                       std::string_view reasoning_effort);

// True when the artifact carries the Jinja template RenderChat mirrors.
[[nodiscard]] bool ValidateGgufTemplate(const core::GgufReader& reader,
                                        std::string* error_msg);

struct ChatRenderOptions {
  /// "low" and "high" are honored; anything else renders as "Max".
  std::string_view reasoning_effort{"max"};
  /// False closes the reasoning block (`<think></think>`) after the prompt.
  bool thinking{true};
  bool add_generation_prompt{true};
};

// Maps the provider-neutral reasoning controls onto the template's low, high
// and max efforts and its open or closed reasoning block.
[[nodiscard]] ChatRenderOptions RenderOptionsFor(
    const ReasoningOptions& reasoning);

// Mirrors the GGUF Jinja template for a whole conversation: system, user,
// assistant (reasoning and tool calls) and tool turns, and the tool list.
// Developer messages render as system messages.
[[nodiscard]] std::string RenderChat(
    std::span<const tokenization::ChatMessage> messages,
    std::span<const tokenization::ChatTool> tools,
    const ChatRenderOptions& options = {});

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_CHAT_TEMPLATE_HPP_
