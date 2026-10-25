// Renders conversations through RenderChat and compares them byte for byte
// with the output of the artifact's Jinja template (fixtures/chat_cases.json,
// produced with jinja2 and Hugging Face's tojson spelling).
#include "src/models/glm53_flash/chat_template.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/json.hpp"

namespace {

namespace tok = gufo::tokenization;
namespace json = gufo::json;

tok::ChatMessage ToMessage(const json::Value& v) {
  tok::ChatMessage m;
  const auto role = v.member_str("role");
  m.role = role == "system"      ? tok::ChatRole::kSystem
           : role == "assistant" ? tok::ChatRole::kAssistant
           : role == "tool"      ? tok::ChatRole::kTool
                                 : tok::ChatRole::kUser;
  m.content = v.member_str("content");
  m.thought = v.member_str("reasoning_content");
  m.tool_call_id = v.member_str("tool_call_id");
  if (const auto* calls = v.find("tool_calls")) {
    for (const auto& c : calls->items()) {
      tok::ChatMessage::ToolCall call;
      call.id = c.member_str("id");
      const auto& fn = *c.find("function");
      call.name = fn.member_str("name");
      for (const auto& [key, value] : fn.find("arguments")->members()) {
        call.arguments.push_back(
            {key, value.is_string() ? value.str() : value.dump(),
             value.is_string()});
      }
      m.tool_calls.push_back(std::move(call));
    }
  }
  return m;
}

tok::ChatTool ToTool(const json::Value& v) {
  tok::ChatTool tool;
  const auto& fn = *v.find("function");
  tool.name = fn.member_str("name");
  tool.description = fn.member_str("description");
  tool.parameters_json = fn.find("parameters")->dump();
  tool.definition_json = v.dump();
  return tool;
}

}  // namespace

int main() {
  std::ifstream in(GUFO_GLM53_CHAT_CASES);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const auto cases = json::parse(buffer.str());
  int failures = 0;
  for (const auto& c : cases.items()) {
    std::vector<tok::ChatMessage> messages;
    for (const auto& m : c.find("messages")->items())
      messages.push_back(ToMessage(m));
    std::vector<tok::ChatTool> tools;
    if (const auto* t = c.find("tools"); t != nullptr && t->is_array()) {
      for (const auto& tool : t->items())
        tools.push_back(ToTool(tool));
    }
    gufo::models::glm53_flash::ChatRenderOptions options;
    const auto effort = c.member_str("effort");
    if (!effort.empty())
      options.reasoning_effort = effort;
    options.add_generation_prompt = c.find("gen")->as_bool(true);
    const auto got =
        gufo::models::glm53_flash::RenderChat(messages, tools, options);
    const auto expected = c.member_str("expected");
    const bool ok = got == expected;
    std::printf("%-28s %s\n", c.member_str("name").c_str(), ok ? "OK" : "FAIL");
    if (!ok) {
      ++failures;
      std::printf("  expected: %s\n  got:      %s\n",
                  json::Value(expected).dump().c_str(),
                  json::Value(got).dump().c_str());
    }
  }
  return failures == 0 ? 0 : 1;
}
