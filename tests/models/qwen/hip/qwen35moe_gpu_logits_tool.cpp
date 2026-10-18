// Development tool for the Qwen GPU MoE path. It runs either a single forward
// pass over a list of token ids to dump the last position's logits, or a greedy
// generation loop, so the GPU MoE path can be compared against the CPU
// reference and the llama.cpp CPU oracle without a full text harness.
//
// usage: qwen35moe_gpu_logits_tool <model.gguf> <ids.txt> <out> [n_gen]
//   without n_gen: writes <out> as raw float32 logits of the last position.
//   with n_gen:    writes <out> as n_gen greedy token ids, space separated.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
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
  if (const char* rows_env = std::getenv("VERIFY_ROWS")) {
    const std::size_t rows = std::stoul(rows_env);
    const std::vector<gufo::tokenization::TokenId> head(ids.begin(),
                                                        ids.end() - rows);
    const auto start = static_cast<std::uint32_t>(head.size());
    std::vector<std::vector<float>> sequential;
    for (int pass = 0; pass < 2; ++pass) {
      sequential.clear();
      executor->Reset();
      (void)executor->ForwardPromptBatch(head, 0, true);
      for (std::size_t row = 0; row < rows; ++row) {
        (void)executor->ForwardToken(ids[head.size() + row],
                                     start + static_cast<std::uint32_t>(row));
        const auto logits = executor->CopyLastLogits();
        sequential.emplace_back(logits.begin(), logits.end());
      }
    }
    executor->Reset();
    (void)executor->ForwardPromptBatch(head, 0, true);
    const gufo::hip::QwenGpuVerificationItem item{
        executor.get(), std::span(ids).subspan(head.size()), start, true};
    std::vector<std::vector<float>> again;
    if (std::getenv("VERIFY_CONTROL")) {
      for (std::size_t row = 0; row < rows; ++row) {
        (void)executor->ForwardToken(ids[head.size() + row],
                                     start + static_cast<std::uint32_t>(row));
        const auto logits = executor->CopyLastLogits();
        again.emplace_back(logits.begin(), logits.end());
      }
    } else {
      (void)gufo::hip::QwenGpuExecutor::ForwardVerificationBatch(
          std::span(&item, 1));
    }
    std::vector<std::vector<float>> oracle;
    for (std::size_t row = 0; row < rows; ++row) {
      const std::vector<gufo::tokenization::TokenId> prefix(
          ids.begin(),
          ids.begin() + static_cast<std::ptrdiff_t>(head.size() + row + 1));
      executor->Reset();
      (void)executor->ForwardPromptBatch(prefix, 0, true);
      const auto logits = executor->CopyLastLogits();
      oracle.emplace_back(logits.begin(), logits.end());
    }
    executor->Reset();
    (void)executor->ForwardPromptBatch(head, 0, true);
    (void)gufo::hip::QwenGpuExecutor::ForwardVerificationBatch(
        std::span(&item, 1));
    for (std::size_t row = 0; row < rows; ++row) {
      double d_seq = 0.0;
      double d_bat = 0.0;
      const auto batched_row = executor->CopyVerificationLogits(row);
      for (std::size_t i = 0; i < oracle[row].size(); ++i) {
        d_seq += std::abs(oracle[row][i] - sequential[row][i]);
        d_bat += std::abs(oracle[row][i] - batched_row[i]);
      }
      std::printf(
          "oracle row %zu: mean|prefill-decode| %.4f mean|prefill-batched| "
          "%.4f\n",
          row, d_seq / static_cast<double>(oracle[row].size()),
          d_bat / static_cast<double>(oracle[row].size()));
    }
    for (std::size_t row = 0; row < rows; ++row) {
      const std::span<const float> batched =
          again.empty() ? executor->CopyVerificationLogits(row)
                        : std::span<const float>(again[row]);
      const auto& ref = sequential[row];
      float max_diff = 0.0F;
      for (std::size_t i = 0; i < ref.size(); ++i)
        max_diff = std::max(max_diff, std::abs(ref[i] - batched[i]));
      const auto top_ref = std::max_element(ref.begin(), ref.end());
      const auto top_bat = std::max_element(batched.begin(), batched.end());
      double sum_diff = 0.0;
      std::size_t worst = 0;
      for (std::size_t i = 0; i < ref.size(); ++i) {
        sum_diff += std::abs(ref[i] - batched[i]);
        if (std::abs(ref[i] - batched[i]) >
            std::abs(ref[worst] - batched[worst]))
          worst = i;
      }
      std::printf(
          "  mean|diff| %.4f worst idx %zu ref %.3f batched %.3f top ref %.3f "
          "batched %.3f\n",
          sum_diff / static_cast<double>(ref.size()), worst, ref[worst],
          batched[worst], *std::max_element(ref.begin(), ref.end()),
          *std::max_element(batched.begin(), batched.end()));
      std::vector<float> sorted(ref);
      std::partial_sort(sorted.begin(), sorted.begin() + 2, sorted.end(),
                        std::greater<>());
      std::printf(
          "row %zu max|diff| %.4f top ref %td batched %td margin %.4f\n", row,
          max_diff, top_ref - ref.begin(), top_bat - batched.begin(),
          sorted[0] - sorted[1]);
    }
    return 0;
  }
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
