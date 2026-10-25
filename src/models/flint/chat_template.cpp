#include "src/models/flint/chat_template.hpp"

namespace gufo::models::flint {
namespace {

// flint_text(): strip control tokens and escape the tag set as HTML entities.
std::string FlintText(std::string_view t) {
  std::string out;
  out.reserve(t.size());
  for (std::size_t i = 0; i < t.size();) {
    auto starts_with = [&](std::string_view needle) {
      return !needle.empty() && t.size() - i >= needle.size() &&
             t.compare(i, needle.size(), needle) == 0;
    };
    if (t[i] == '\0') {
      ++i;
      continue;
    }
    if (starts_with(kThinkOpen)) {
      i += kThinkOpen.size();
      continue;
    }
    if (starts_with(kThinkClose)) {
      i += kThinkClose.size();
      continue;
    }
    struct Escaped {
      std::string_view from;
      std::string_view to;
    };
    static constexpr Escaped kEsc[] = {
        {"<|begin_of_text|>", "&lt;|begin_of_text|&gt;"},
        {"<|end_of_text|>", "&lt;|end_of_text|&gt;"},
        {"<|system|>", "&lt;|system|&gt;"},
        {"<|developer|>", "&lt;|developer|&gt;"},
        {"<|user|>", "&lt;|user|&gt;"},
        {"<|assistant|>", "&lt;|assistant|&gt;"},
        {"<|observation|>", "&lt;|observation|&gt;"},
    };
    bool replaced = false;
    for (const auto& e : kEsc) {
      if (starts_with(e.from)) {
        out.append(e.to);
        i += e.from.size();
        replaced = true;
        break;
      }
    }
    if (!replaced) {
      out.push_back(t[i]);
      ++i;
    }
  }
  return out;
}

}  // namespace

std::string RenderPrompt(const std::vector<ChatMessage>& messages,
                         const ChatTemplateOptions& options) {
  (void)options;
  std::string out;
  for (const auto& m : messages) {
    if (m.role == "system") {
      out += kSystemTag;
      out += FlintText(m.content);
    } else if (m.role == "developer") {
      out += "<|developer|>";
      out += FlintText(m.content);
    } else if (m.role == "user") {
      out += kUserTag;
      out += FlintText(m.content);
    } else if (m.role == "assistant") {
      out += kAssistantTag;
      out += "\n";
      if (!m.reasoning_content.empty()) {
        out += kThinkOpen;
        out += FlintText(m.reasoning_content);
        out += kThinkClose;
      }
      out += FlintText(m.content);
    } else if (m.role == "tool") {
      out += "<|observation|>";
      out += "<tool_response>";
      out += FlintText(m.content);
      out += "</tool_response>";
    }
  }
  // add_generation_prompt: only the assistant tag; jinja whitespace control
  // strips surrounding newlines.
  out += kAssistantTag;
  return out;
}

std::string RenderPrompt(std::string_view system, std::string_view user,
                         const ChatTemplateOptions& options) {
  std::vector<ChatMessage> messages;
  if (!system.empty()) {
    messages.push_back({"system", std::string(system), ""});
  }
  messages.push_back({"user", std::string(user), ""});
  return RenderPrompt(messages, options);
}

}  // namespace gufo::models::flint
