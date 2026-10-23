#ifndef GUFO_MODELS_GLM53_FLASH_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_GLM53_FLASH_CHAT_TEMPLATE_HPP_

#include <string>
#include <string_view>

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

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_CHAT_TEMPLATE_HPP_
