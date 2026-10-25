// Cycle cost of a speculative step by forced draft width and context depth:
//   glm53_cost_audit MODEL DFLASH TOKENS.txt [depths=0,4096,32768] [steps=12]
// TOKENS.txt supplies the text; the context is its tokens repeated to depth.
// Prints the median milliseconds per step and the mean accepted drafts for
// widths 0 (ordinary step) to 7.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "src/models/glm53_flash/engine.hpp"

namespace {

namespace glm = gufo::models::glm53_flash;
constexpr int kSkip = 77;
constexpr int kMaxWidth = 7;

std::vector<std::uint32_t> ParseList(const std::string& text) {
  std::vector<std::uint32_t> out;
  std::stringstream in(text);
  std::string item;
  while (std::getline(in, item, ','))
    out.push_back(static_cast<std::uint32_t>(std::atol(item.c_str())));
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::printf("usage: %s MODEL DFLASH TOKENS.txt [depths] [steps]\n",
                argv[0]);
    return kSkip;
  }
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::vector<std::int32_t> text;
  {
    std::ifstream in(argv[3]);
    long v;
    while (in >> v)
      text.push_back(static_cast<std::int32_t>(v));
  }
  const auto depths = ParseList(argc > 4 ? argv[4] : "0,4096,32768");
  const int steps = argc > 5 ? std::atoi(argv[5]) : 12;
  if (text.empty() || depths.empty() || steps < 1) {
    std::printf("empty input\n");
    return 1;
  }
  const std::uint32_t deepest = *std::max_element(depths.begin(), depths.end());

  std::string error;
  glm::ModelOptions options;
  options.max_context = deepest + 1024;
  options.dflash_model_path = argv[2];
  auto model = glm::Model::Load(argv[1], options, &error);
  if (!model) {
    std::printf("load: %s\n", error.c_str());
    return 1;
  }
  std::printf("depth width   ms/step  accepted/step\n");
  for (const auto depth : depths) {
    std::vector<std::int32_t> prompt;
    const std::uint32_t length = std::max<std::uint32_t>(depth, 64);
    for (std::uint32_t i = 0; i < length; ++i)
      prompt.push_back(text[i % text.size()]);
    for (int width = 0; width <= kMaxWidth; ++width) {
      auto session = model->CreateSession(gufo::core::SessionMode::kSpeculative,
                                          options.max_context, &error);
      if (!session || !session->Sync(prompt, &error)) {
        std::printf("session: %s\n", error.c_str());
        return 1;
      }
      session->ForceDraftWidth(width);
      gufo::sampling::SamplerState sampler;
      std::vector<double> ms;
      std::uint64_t accepted = 0;
      for (int i = 0; i < steps + 2; ++i) {
        glm::Session::DecodeResult step;
        const auto before = session->Statistics().accepted;
        const auto start = std::chrono::steady_clock::now();
        if (!session->DecodeStep(kMaxWidth + 1, sampler, &step, &error,
                                 false)) {
          std::printf("decode: %s\n", error.c_str());
          return 1;
        }
        const double elapsed = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        if (i >= 2) {
          ms.push_back(elapsed);
          accepted += session->Statistics().accepted - before;
        }
      }
      std::sort(ms.begin(), ms.end());
      std::printf("%5u %5d %9.2f %9.2f\n", depth, width, ms[ms.size() / 2],
                  static_cast<double>(accepted) / steps);
    }
  }
  return 0;
}
