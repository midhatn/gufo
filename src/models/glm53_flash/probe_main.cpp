#include <cstdio>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/runtime/config.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <model-00001-of-N.gguf>\n", argv[0]);
    return 2;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 1;
  }
  auto config =
      gufo::models::glm53_flash::ParseConfig(*reader, &error);
  if (!config) {
    std::fprintf(stderr, "config rejected: %s\n", error.c_str());
    return 1;
  }
  const auto& c = *config;
  std::printf(
      "layers=%u trunk=%u dense=%u mla=%u kda=%u hidden=%u heads=%u\n"
      "kv_lora=%u q_lora=%u mla_head=%u/%u indexer=%u x %u top_k=%u "
      "kpool=%u\n"
      "experts=%u used=%u shared=%u ff=%u/%u dense_ff=%u\n"
      "hc=%u rank=%u sinkhorn=%u kda_dim=%u conv=%u vocab=%u ctx=%u "
      "nextn=%u\n"
      "eps=%g/%g/%g gate_lb=%g weights_scale=%g norm=%d\n"
      "tensors=%zu\n",
      c.num_layers, c.TrunkLayers(), c.leading_dense, c.NumMlaLayers(),
      c.NumKdaLayers(), c.hidden_size, c.num_heads, c.kv_lora_rank,
      c.q_lora_rank, c.key_length_mla, c.value_length_mla, c.indexer_heads,
      c.indexer_key_dim, c.indexer_top_k, c.indexer_kpool, c.expert_count,
      c.expert_used, c.expert_shared, c.expert_ff, c.shared_ff, c.dense_ff,
      c.hc_count, c.HcRank(), c.sinkhorn_iters, c.KdaDim(),
      c.ssm_conv_kernel, c.vocab_size, c.context_length, c.nextn_layers,
      c.rms_eps, c.layer_norm_eps, c.hc_eps, c.kda_gate_lower_bound,
      c.expert_weights_scale, c.expert_weights_norm ? 1 : 0,
      reader->GetTensors().size());
  for (std::uint32_t layer = 0; layer < c.num_layers; ++layer) {
    std::printf("blk.%02u %s%s%s\n", layer,
                c.IsMla(layer) ? "mla" : "kda",
                layer >= c.leading_dense ? "+moe" : "+dense",
                layer == c.MtpLayer() ? " mtp" : "");
  }
  return 0;
}
