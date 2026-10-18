// Development tool: runs the CPU reference forward pass of a Qwen GGUF over a
// list of token ids and writes the last position's logits as raw float32.
// usage: qwen35moe_logits_tool <model.gguf> <ids.txt> <logits.bin>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/ssm.hpp"
#include "src/models/qwen/state.hpp"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << " model.gguf ids.txt logits.bin\n";
    return 2;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::cerr << "open failed: " << error << '\n';
    return 1;
  }
  auto weights = gufo::models::QwenModelWeights::LoadFromGguf(*reader, &error);
  if (!weights) {
    std::cerr << "weights failed: " << error << '\n';
    return 1;
  }
  std::vector<std::uint32_t> ids;
  std::ifstream in(argv[2]);
  for (std::uint32_t id; in >> id;)
    ids.push_back(id);
  const auto& c = weights->config;
  gufo::models::QwenKvCache kv(c.FullAttentionLayerCount(),
                               c.num_key_value_heads,
                               static_cast<std::uint32_t>(ids.size() + 1),
                               c.head_dim);
  gufo::models::QwenSsmCache ssm(c.num_layers, c.SsmQkvSize(),
                                 c.ssm_conv_kernel, c.ssm_time_step_rank,
                                 c.ssm_state_size, c.SsmValueSize());
  gufo::models::QwenScratchArena arena(c);
  for (std::size_t p = 0; p < ids.size(); ++p) {
    gufo::models::ForwardModel(ids[p], static_cast<std::uint32_t>(p), *weights,
                               kv, ssm, arena, arena.logits);
    std::fprintf(stderr, "pos %zu/%zu\r", p + 1, ids.size());
  }
  std::ofstream out(argv[3], std::ios::binary);
  out.write(reinterpret_cast<const char*>(arena.logits.data()),
            static_cast<std::streamsize>(arena.logits.size() * sizeof(float)));
  std::fprintf(stderr, "\ndone\n");
  return 0;
}
