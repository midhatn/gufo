// Development tool for the Qwen GPU MoE path. It runs either a single forward
// pass over a list of token ids to dump the last position's logits, or a greedy
// generation loop, so the GPU MoE path can be compared against the CPU
// reference and the llama.cpp CPU oracle without a full text harness.
//
// usage: qwen35moe_gpu_logits_tool <model.gguf> <ids.txt> <out> [n_gen]
//   without n_gen: writes <out> as raw float32 logits of the last position.
//   with n_gen:    writes <out> as n_gen greedy token ids, space separated.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cerr << "usage: " << argv[0]
              << " model.gguf ids.txt out.bin [n_gen]\n";
    return 2;
  }
  gufo::models::qwen38_flash_next::rocm::RegisterQwenMoePrefill();
  const bool generate = argc == 5;
  std::string error;
  auto reader_owner = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader_owner) {
    std::cerr << "open failed: " << error << '\n';
    return 1;
  }
  const std::shared_ptr<const gufo::core::GgufReader> reader(
      std::move(reader_owner));
  auto executor = gufo::hip::QwenGpuExecutor::CreateFromGguf(reader, &error);
  if (!executor) {
    std::cerr << "executor failed: " << error << '\n';
    return 1;
  }
  std::vector<gufo::tokenization::TokenId> ids;
  std::ifstream in(argv[2]);
  for (std::uint32_t id; in >> id;)
    ids.push_back(id);
  (void)executor->ForwardPromptBatch(ids, 0, true);

  if (!generate) {
    const auto logits = executor->CopyLastLogits();
    std::ofstream out(argv[3], std::ios::binary);
    out.write(reinterpret_cast<const char*>(logits.data()),
              static_cast<std::streamsize>(logits.size() * sizeof(float)));
    std::fprintf(stderr, "wrote %zu logits\n", logits.size());
    return 0;
  }

  const int n_gen = std::stoi(argv[4]);
  std::uint32_t pos = static_cast<std::uint32_t>(ids.size());
  std::vector<gufo::tokenization::TokenId> generated;
  for (int i = 0; i < n_gen; ++i) {
    const auto logits = executor->CopyLastLogits();
    const auto best = std::max_element(logits.begin(), logits.end());
    const auto next = static_cast<gufo::tokenization::TokenId>(
        std::distance(logits.begin(), best));
    generated.push_back(next);
    if (i + 1 < n_gen) {
      (void)executor->ForwardToken(next, pos++);
    }
  }
  std::ofstream out(argv[3]);
  for (std::size_t i = 0; i < generated.size(); ++i) {
    out << (i == 0 ? "" : " ") << generated[i];
  }
  out << '\n';
  std::fprintf(stderr, "wrote %zu tokens\n", generated.size());
  return 0;
}
