#include "src/models/glm53_flash/chat_template.hpp"

namespace gufo::models::glm53_flash {

std::string RenderPrompt(std::string_view system, std::string_view user,
                         std::string_view reasoning_effort) {
  std::string out = "[gMASK]<sop>";
  out += kSystemTag;
  out += "Reasoning Effort: ";
  if (reasoning_effort == "low") {
    out += "Low";
  } else if (reasoning_effort == "high") {
    out += "High";
  } else {
    out += "Max";
  }
  if (!system.empty()) {
    out += kSystemTag;
    out += system;
  }
  out += kUserTag;
  out += user;
  out += kAssistantTag;
  out += kThinkOpen;
  return out;
}

}  // namespace gufo::models::glm53_flash
