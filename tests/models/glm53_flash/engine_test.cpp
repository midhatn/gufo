// Engine checks on the real artifact:
//   glm53_engine_test MODEL [DFLASH]
// Snapshot round trips continue bit-identically, a prefix extension matches a
// cold prompt, and seeded sampled decoding with drafts replays exactly.
#include "src/models/glm53_flash/engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

namespace glm = gufo::models::glm53_flash;
constexpr int kSkip = 77;

std::vector<std::int32_t> MakePrompt(const glm::Model& model, std::size_t n) {
  std::string text = "[gMASK]<sop><|user|>";
  for (int i = 0; text.size() < n * 5; ++i) {
    text += "Item " + std::to_string(i * 7919 % 1000) +
            " of the quarterly report covers region " + std::to_string(i % 13) +
            ", where sales grew " + std::to_string(i * 31 % 17) + " percent. ";
  }
  auto tokens = model.Tokenize(text);
  tokens.resize(n);
  return tokens;
}

std::vector<std::int32_t> Greedy(glm::Session& session, std::size_t count) {
  std::vector<std::int32_t> out;
  gufo::sampling::SamplerState sampler;
  std::string error;
  while (out.size() < count) {
    glm::Session::DecodeResult step;
    if (!session.DecodeStep(1, sampler, &step, &error, false)) {
      std::printf("decode: %s\n", error.c_str());
      std::exit(1);
    }
    out.insert(out.end(), step.tokens.begin(), step.tokens.end());
  }
  return out;
}

bool SameLogits(const glm::Session& a, const glm::Session& b) {
  const auto x = a.Logits();
  const auto y = b.Logits();
  return x.size() == y.size() && !x.empty() &&
         std::memcmp(x.data(), y.data(), x.size_bytes()) == 0;
}

// Control tokens in prompt text are single tokens, as in llama.cpp.
bool ControlTokens(const glm::Model& model) {
  const auto framing = model.Tokenize(
      "[gMASK]<sop><|user|>x<|assistant|><think></think><tool_call>f"
      "<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>");
  const auto count = [&](std::int32_t id) {
    return std::count(framing.begin(), framing.end(), id);
  };
  const bool ok =
      framing.size() >= 2 && framing[0] == 154822 && framing[1] == 154824 &&
      count(154827) == 1 && count(154828) == 1 && count(154841) == 1 &&
      count(154842) == 1 && count(154843) == 1 && count(154847) == 1 &&
      count(154848) == 1 && count(154849) == 1 && count(154850) == 1 &&
      count(154844) == 1;
  std::printf("control tokens: %s\n", ok ? "OK" : "FAIL");
  return ok;
}

bool RoundTrip(glm::Model& model, std::size_t n) {
  std::string error;
  const auto prompt = MakePrompt(model, n);
  auto a = model.CreateSession(gufo::core::SessionMode::kAutoregressive,
                               model.MaxContext(), &error);
  auto b = model.CreateSession(gufo::core::SessionMode::kAutoregressive,
                               model.MaxContext(), &error);
  if (!a || !b || !a->Sync(prompt, &error)) {
    std::printf("setup: %s\n", error.c_str());
    return false;
  }
  const auto snapshot = a->SaveSnapshot(&error);
  if (!snapshot) {
    std::printf("snapshot: %s\n", error.c_str());
    return false;
  }
  const auto first = Greedy(*a, 24);
  if (!b->RestoreSnapshot(*snapshot, &error)) {
    std::printf("restore: %s\n", error.c_str());
    return false;
  }
  const auto second = Greedy(*b, 24);
  const bool fresh = first == second && SameLogits(*a, *b);
  // Back into the session that moved on.
  if (!a->RestoreSnapshot(*snapshot, &error)) {
    std::printf("restore: %s\n", error.c_str());
    return false;
  }
  const auto third = Greedy(*a, 24);
  const bool same = first == third && SameLogits(*a, *b);
  std::printf("round trip %zu tokens: %s (%llu byte snapshot)\n", n,
              fresh && same ? "OK" : "FAIL",
              static_cast<unsigned long long>(snapshot->SizeBytes()));
  return fresh && same;
}

bool PrefixExtension(glm::Model& model) {
  std::string error;
  const auto prompt = MakePrompt(model, 301);
  auto warm = model.CreateSession(gufo::core::SessionMode::kAutoregressive,
                                  model.MaxContext(), &error);
  auto cold = model.CreateSession(gufo::core::SessionMode::kAutoregressive,
                                  model.MaxContext(), &error);
  const std::span<const std::int32_t> all(prompt);
  if (!warm->Sync(all.first(200), &error) || !warm->Sync(all, &error) ||
      !cold->Sync(all, &error)) {
    std::printf("sync: %s\n", error.c_str());
    return false;
  }
  const auto x = warm->Logits();
  const auto y = cold->Logits();
  double dot = 0, xx = 0, yy = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    dot += static_cast<double>(x[i]) * y[i];
    xx += static_cast<double>(x[i]) * x[i];
    yy += static_cast<double>(y[i]) * y[i];
  }
  const double cosine = dot / std::sqrt(xx * yy);
  const bool top1 = std::max_element(x.begin(), x.end()) - x.begin() ==
                    std::max_element(y.begin(), y.end()) - y.begin();
  std::printf("prefix extension: cos %.6f top1 %s\n", cosine,
              top1 ? "same" : "DIFFERENT");
  return cosine > 0.999 && top1;
}

bool SeededReplay(glm::Model& model) {
  std::string error;
  const auto prompt = MakePrompt(model, 400);
  std::vector<std::int32_t> runs[2];
  for (auto& run : runs) {
    auto session = model.CreateSession(gufo::core::SessionMode::kSpeculative,
                                       model.MaxContext(), &error);
    if (!session || !session->Sync(prompt, &error)) {
      std::printf("setup: %s\n", error.c_str());
      return false;
    }
    gufo::sampling::SamplingConfig config;
    config.temperature = 0.8F;
    config.top_p = 0.95F;
    config.seed = 1234;
    gufo::sampling::SamplerState sampler(config);
    while (run.size() < 96) {
      glm::Session::DecodeResult step;
      if (!session->DecodeStep(96 - run.size(), sampler, &step, &error)) {
        std::printf("decode: %s\n", error.c_str());
        return false;
      }
      run.insert(run.end(), step.tokens.begin(), step.tokens.end());
    }
  }
  std::printf("seeded replay: %s\n", runs[0] == runs[1] ? "OK" : "FAIL");
  return runs[0] == runs[1];
}

}  // namespace

double FreeGiB() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  hipMemGetInfo(&free_bytes, &total_bytes);
  return static_cast<double>(free_bytes) / (1 << 30);
}

// Prints what a session costs next to the resident model, in GiB.
void SessionFootprint(glm::Model& model) {
  std::string error;
  const double base = FreeGiB();
  auto autoregressive = model.CreateSession(
      gufo::core::SessionMode::kAutoregressive, model.MaxContext(), &error);
  const double after_ar = FreeGiB();
  auto speculative = model.CreateSession(gufo::core::SessionMode::kSpeculative,
                                         model.MaxContext(), &error);
  const double after_spec = FreeGiB();
  const auto prompt = MakePrompt(model, 200);
  if (speculative && speculative->Sync(prompt, &error)) {
    gufo::sampling::SamplerState sampler;
    glm::Session::DecodeResult step;
    for (int i = 0; i < 6; ++i)
      (void)speculative->DecodeStep(8, sampler, &step, &error, false);
  }
  const double after_steps = FreeGiB();
  std::printf(
      "footprint: autoregressive %.2f, speculative %.2f, +after decode %.2f "
      "GiB (estimate %.2f / %.2f)\n",
      base - after_ar, after_ar - after_spec, after_spec - after_steps,
      model.SessionBytes(gufo::core::SessionMode::kAutoregressive,
                         model.MaxContext()) /
          double(1 << 30),
      model.SessionBytes(gufo::core::SessionMode::kSpeculative,
                         model.MaxContext()) /
          double(1 << 30));
}

int main(int argc, char** argv) {
  if (argc < 2 || argv[1][0] == '\0') {
    return kSkip;
  }
  std::string error;
  glm::ModelOptions options;
  options.max_context = 8192;
  if (argc > 2) {
    options.dflash_model_path = argv[2];
  }
  auto model = glm::Model::Load(argv[1], options, &error);
  if (!model) {
    std::printf("load: %s\n", error.c_str());
    return kSkip;
  }
  SessionFootprint(*model);
  bool ok = ControlTokens(*model);
  ok &= RoundTrip(*model, 301);
  ok &= RoundTrip(*model, 4103);
  ok &= PrefixExtension(*model);
  if (model->HasDflash()) {
    ok &= SeededReplay(*model);
  }
  std::printf(ok ? "OK\n" : "FAIL\n");
  return ok ? 0 : 1;
}
