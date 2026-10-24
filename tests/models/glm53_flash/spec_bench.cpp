// Plain greedy decode versus DFlash-2 plus history drafting on one prompt:
//   glm53_spec_bench MODEL DFLASH TOKENS.txt [max_new]
// Prints decode speed, draft statistics, and how far the two outputs agree.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "src/models/glm53_flash/engine.hpp"

namespace {

namespace glm = gufo::models::glm53_flash;
constexpr int kSkip = 77;

double Seconds(std::chrono::steady_clock::time_point from) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - from)
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::printf("usage: %s MODEL DFLASH TOKENS.txt [max_new]\n", argv[0]);
    return kSkip;
  }
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::vector<std::int32_t> prompt;
  {
    std::ifstream in(argv[3]);
    long v;
    while (in >> v)
      prompt.push_back(static_cast<std::int32_t>(v));
  }
  const std::size_t max_new = argc > 4 ? std::atoi(argv[4]) : 128;

  std::string error;
  glm::ModelOptions options;
  options.max_context =
      static_cast<std::uint32_t>(prompt.size() + max_new + 64);
  options.dflash_model_path = argv[2];
  auto model = glm::Model::Load(argv[1], options, &error);
  if (!model) {
    std::printf("load: %s\n", error.c_str());
    return 1;
  }

  const auto generate = [&](gufo::core::SessionMode mode, std::size_t per_step,
                            std::vector<std::int32_t>* out, double* seconds,
                            glm::Session::SpeculativeStats* stats) {
    auto session = model->CreateSession(mode, options.max_context, &error);
    if (!session || !session->Sync(prompt, &error)) {
      std::printf("session: %s\n", error.c_str());
      return false;
    }
    gufo::sampling::SamplerState sampler;
    const auto start = std::chrono::steady_clock::now();
    while (out->size() < max_new) {
      glm::Session::DecodeResult step;
      if (!session->DecodeStep(std::min(per_step, max_new - out->size()),
                               sampler, &step, &error, false)) {
        std::printf("decode: %s\n", error.c_str());
        return false;
      }
      out->insert(out->end(), step.tokens.begin(), step.tokens.end());
      if (step.stop)
        break;
    }
    *seconds = Seconds(start);
    *stats = session->Statistics();
    return true;
  };

  std::vector<std::int32_t> plain;
  double plain_s = 0;
  glm::Session::SpeculativeStats plain_stats;
  if (!generate(gufo::core::SessionMode::kAutoregressive, 1, &plain, &plain_s,
                &plain_stats)) {
    return 1;
  }
  std::printf("plain  : %zu tokens in %.2f s = %.1f tok/s\n", plain.size(),
              plain_s, plain.size() / plain_s);

  std::vector<std::int32_t> spec;
  double spec_s = 0;
  glm::Session::SpeculativeStats st;
  if (!generate(gufo::core::SessionMode::kSpeculative, max_new, &spec, &spec_s,
                &st)) {
    return 1;
  }
  std::printf("spec   : %zu tokens in %.2f s = %.1f tok/s\n", spec.size(),
              spec_s, spec.size() / spec_s);
  std::printf(
      "         steps %llu, drafted %llu, accepted %llu (%.2f/step), "
      "emitted/step %.2f, history steps %llu\n",
      static_cast<unsigned long long>(st.cycles),
      static_cast<unsigned long long>(st.drafted),
      static_cast<unsigned long long>(st.accepted),
      st.cycles ? double(st.accepted) / st.cycles : 0.0,
      st.cycles ? double(spec.size()) / st.cycles : 0.0,
      static_cast<unsigned long long>(st.history_cycles));
  std::size_t same = 0;
  while (same < std::min(plain.size(), spec.size()) &&
         plain[same] == spec[same]) {
    ++same;
  }
  std::printf("outputs agree for the first %zu of %zu tokens\n", same,
              std::min(plain.size(), spec.size()));
  return 0;
}
