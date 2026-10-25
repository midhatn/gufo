// Full-trunk check for the production ReferenceModel: evaluates the pinned
// six-token prompt token by token and compares the top-5 logits against the
// goldens captured from the validated reference harness. Loads the entire
// model; run only when no other model is resident (GUFO_GLM53_ALLOW_FULL=1).
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/glm53_flash/reference.hpp"

namespace {
constexpr std::int32_t kTokens[] = {9703, 11, 1879, 0, 101726, 98384};
}  // namespace

int main(int argc, char** argv) {
  if (std::getenv("GUFO_GLM53_ALLOW_FULL") == nullptr) {
    std::cerr << "set GUFO_GLM53_ALLOW_FULL=1 to run (loads the full model)\n";
    return 77;
  }
  const char* goldens = std::getenv("GUFO_GLM53_GOLDENS");
  if (goldens == nullptr || argc < 2) {
    std::cerr << "usage: glm53_reference_check MODEL.GGUF "
                 "(GUFO_GLM53_GOLDENS=dir)\n";
    return 77;
  }
  std::string error;
  auto model = gufo::models::glm53_flash::ReferenceModel::Open(argv[1], &error);
  if (!model) {
    std::cerr << "open: " << error << "\n";
    return 1;
  }
  auto session = model->CreateSession();
  bool ok = true;
  for (std::size_t t = 0; t + 1 < std::size(kTokens); ++t) {
    const std::vector<float>& logits = model->Evaluate(*session, kTokens[t]);
    std::vector<float> top_logit(5, -1e30F);
    std::uint32_t top_id[5] = {0, 0, 0, 0, 0};
    for (std::uint64_t i = 0; i < logits.size(); ++i) {
      if (logits[i] > top_logit[4]) {
        int k = 4;
        while (k > 0 && top_logit[k - 1] < logits[i]) {
          top_logit[k] = top_logit[k - 1];
          top_id[k] = top_id[k - 1];
          --k;
        }
        top_logit[k] = logits[i];
        top_id[k] = static_cast<std::uint32_t>(i);
      }
    }
    std::string path(goldens);
    path += "/logits_top5." + std::to_string(t) + ".f32";
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      std::cerr << "missing golden " << path << "\n";
      return 77;
    }
    float ref[10];
    in.read(reinterpret_cast<char*>(ref), sizeof(ref));
    std::uint32_t ref_id;
    int mismatches = 0;
    for (int k = 0; k < 5; ++k) {
      std::memcpy(&ref_id, &ref[2 * k + 1], 4);
      const std::uint32_t id = top_id[k];
      const float diff = std::abs(top_logit[k] - ref[2 * k]);
      if (id != ref_id || diff > 1e-3F) {
        std::cout << "t=" << t << " slot " << k << ": got " << top_logit[k]
                  << "/" << id << " want " << ref[2 * k] << "/" << ref_id
                  << "\n";
        ++mismatches;
      }
    }
    std::cout << "logits t=" << t << (mismatches ? " MISMATCH" : " OK") << "\n";
    ok = ok && mismatches == 0;
  }
  return ok ? 0 : 1;
}
