// Plain greedy decode versus DFlash-2 speculative decode on one prompt:
//   glm53_spec_bench MODEL DFLASH TOKENS.txt [max_new]
// Prints decode speed, draft acceptance, and how far the two outputs agree.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "src/models/glm53_flash/kernels/rocm/executor.hpp"
#include "src/models/glm53_flash/speculative.hpp"

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
  glm::Speculator::DrafterInfo info;
  if (!glm::Speculator::ReadDrafterInfo(argv[2], &info, &error)) {
    std::printf("drafter info: %s\n", error.c_str());
    return 1;
  }
  glm::rocm::ExecutorOptions options;
  options.max_context =
      static_cast<std::uint32_t>(prompt.size() + max_new + 64);
  options.dflash_layers = info.target_layers;
  options.dflash_rows = options.prefill_chunk;
  options.logit_rows = info.block_size;
  auto executor = glm::rocm::Executor::Create(argv[1], options, &error);
  if (!executor) {
    std::printf("executor: %s\n", error.c_str());
    return 1;
  }

  // Plain greedy decode.
  std::vector<std::int32_t> plain;
  {
    auto session = executor->CreateSession(&error);
    if (!executor->Prefill(*session, prompt, &error)) {
      std::printf("prefill: %s\n", error.c_str());
      return 1;
    }
    std::int32_t token = executor->ArgmaxRows(1)[0];
    plain.push_back(token);
    const auto start = std::chrono::steady_clock::now();
    while (plain.size() < max_new) {
      if (!executor->Evaluate(*session, token, &error)) {
        std::printf("evaluate: %s\n", error.c_str());
        return 1;
      }
      token = executor->ArgmaxRows(1)[0];
      plain.push_back(token);
    }
    const double t = Seconds(start);
    std::printf("plain  : %zu tokens in %.2f s = %.1f tok/s\n",
                plain.size() - 1, t, (plain.size() - 1) / t);
  }

  executor->PrintProfile();
  auto speculator =
      glm::Speculator::Create(*executor, argv[2], options.max_context, &error);
  if (!speculator) {
    std::printf("speculator: %s\n", error.c_str());
    return 1;
  }
  std::vector<std::int32_t> spec;
  {
    auto session = executor->CreateSession(&error);
    const auto start = std::chrono::steady_clock::now();
    const bool ok = speculator->Generate(
        *session, prompt, max_new, {},
        [&](std::int32_t t) {
          spec.push_back(t);
          return true;
        },
        &error);
    const double t = Seconds(start);
    if (!ok) {
      std::printf("generate: %s\n", error.c_str());
      return 1;
    }
    const auto& st = speculator->stats();
    std::printf("spec   : %zu tokens (incl. prefill) in %.2f s\n", spec.size(),
                t);
    std::printf(
        "         steps %llu, drafted %llu, accepted %llu (%.2f/step), "
        "emitted/step %.2f, replays %llu\n",
        static_cast<unsigned long long>(st.steps),
        static_cast<unsigned long long>(st.drafted),
        static_cast<unsigned long long>(st.accepted),
        st.steps ? double(st.accepted) / st.steps : 0.0,
        st.steps ? double(st.emitted) / st.steps : 0.0,
        static_cast<unsigned long long>(st.replays));
  }
  std::printf("speculative stages:\n");
  executor->PrintProfile();
  {
    const auto& st = speculator->stats();
    std::printf("         accepted-per-step histogram:");
    for (int i = 0; i < 8; ++i) {
      std::printf(" %d:%llu", i,
                  static_cast<unsigned long long>(st.accepted_histogram[i]));
    }
    std::printf("\n");
  }
  std::size_t same = 0;
  while (same < std::min(plain.size(), spec.size()) &&
         plain[same] == spec[same]) {
    ++same;
  }
  std::printf("outputs agree for the first %zu of %zu tokens\n", same,
              std::min(plain.size(), spec.size()));
  return 0;
}
