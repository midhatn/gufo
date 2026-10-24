
# GLM-5.3-Flash CTest checks. Artifact-backed tests skip (77) without
# GUFO_GLM53_FLASH_MODEL so hosted CI stays green.
set(GUFO_GLM53_FLASH_MODEL "$ENV{GUFO_GLM53_FLASH_MODEL}" CACHE FILEPATH
  "GLM-5.3-Flash GGUF shard 1 for external-model tests")
add_executable(glm53_chat_template_test chat_template_test.cpp)
target_link_libraries(glm53_chat_template_test PRIVATE gufo_glm53_flash)
target_compile_definitions(glm53_chat_template_test PRIVATE
  GUFO_GLM53_CHAT_CASES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/chat_cases.json")
add_test(NAME glm53.chat_template COMMAND glm53_chat_template_test)
set_tests_properties(glm53.chat_template PROPERTIES LABELS "cpu;glm53;tokenization")

add_executable(glm53_layout_test
  layout_test.cpp
  ${PROJECT_SOURCE_DIR}/src/models/glm53_flash/runtime/config.cpp
  ${PROJECT_SOURCE_DIR}/src/models/glm53_flash/runtime/manifest.cpp
  ${PROJECT_SOURCE_DIR}/src/models/glm53_flash/runtime/layout.cpp)
target_link_libraries(glm53_layout_test PRIVATE gufo_core)
add_test(NAME glm53.layout COMMAND glm53_layout_test)
set_tests_properties(glm53.layout PROPERTIES
  LABELS "cpu;glm53;loader;external-model"
  SKIP_RETURN_CODE 77)

# Leading 3-layer KDA forward on the real artifact; dumps every stage.
add_executable(glm53_forward_layer0_test
  forward_layer0_test.cpp
  ${PROJECT_SOURCE_DIR}/src/models/glm53_flash/runtime/config.cpp
  ${PROJECT_SOURCE_DIR}/src/models/glm53_flash/runtime/manifest.cpp
  ${PROJECT_SOURCE_DIR}/src/models/glm53_flash/runtime/layout.cpp)
target_link_libraries(glm53_forward_layer0_test PRIVATE gufo_core)
add_test(NAME glm53.forward0
  COMMAND glm53_forward_layer0_test "${GUFO_GLM53_FLASH_MODEL}")
set_tests_properties(glm53.forward0 PROPERTIES
  LABELS "cpu;glm53;reference;external-model"
  SKIP_RETURN_CODE 77
  ENVIRONMENT "GUFO_GLM53_DUMP_DIR=${CMAKE_CURRENT_BINARY_DIR}/glm53-dump;GUFO_GLM53_LAYERS=4"
  FIXTURES_SETUP glm53-dump
  COST 300)

find_program(PYTHON3_EXECUTABLE python3)

# DSA sparsity unit: 48-token synthetic stream, top-8 pool selection.
add_test(NAME glm53.dsasp
  COMMAND glm53_forward_layer0_test "${GUFO_GLM53_FLASH_MODEL}")
set_tests_properties(glm53.dsasp PROPERTIES
  LABELS "cpu;glm53;reference;external-model"
  SKIP_REGULAR_EXPRESSION "model artifact unavailable"
  ENVIRONMENT "GUFO_GLM53_DUMP_DIR=${CMAKE_CURRENT_BINARY_DIR}/glm53-sp;GUFO_GLM53_DSASP=8"
  FIXTURES_SETUP glm53-sp
  COST 120)

if(PYTHON3_EXECUTABLE)
  add_test(NAME glm53.dsasp_numpy
    COMMAND ${PYTHON3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/layer0_numpy.py
      "${GUFO_GLM53_FLASH_MODEL}"
      ${CMAKE_CURRENT_BINARY_DIR}/glm53-sp
      --dsasp)
  set_tests_properties(glm53.dsasp_numpy PROPERTIES
    LABELS "cpu;glm53;reference;external-model"
    SKIP_RETURN_CODE 77
    FIXTURES_REQUIRED glm53-sp
    ENVIRONMENT "GUFO_GLM53_TOPPOOL=8"
    COST 120)
endif()

# Teacher-forcing logits cross-check through 8 layers (slow; -LE slow to skip).
add_test(NAME glm53.logits8
  COMMAND glm53_forward_layer0_test "${GUFO_GLM53_FLASH_MODEL}")
set_tests_properties(glm53.logits8 PROPERTIES
  LABELS "cpu;glm53;reference;external-model;slow"
  SKIP_REGULAR_EXPRESSION "model artifact unavailable"
  ENVIRONMENT "GUFO_GLM53_DUMP_DIR=${CMAKE_CURRENT_BINARY_DIR}/glm53-dump8;GUFO_GLM53_LAYERS=8;GUFO_GLM53_LOGITS=1"
  FIXTURES_SETUP glm53-dump8
  COST 900)

if(PYTHON3_EXECUTABLE)
  add_test(NAME glm53.logits8_numpy
    COMMAND ${PYTHON3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/layer0_numpy.py
      "${GUFO_GLM53_FLASH_MODEL}"
      ${CMAKE_CURRENT_BINARY_DIR}/glm53-dump8)
  set_tests_properties(glm53.logits8_numpy PROPERTIES
    LABELS "cpu;glm53;reference;external-model;slow"
    SKIP_RETURN_CODE 77
    FIXTURES_REQUIRED glm53-dump8
    ENVIRONMENT "GUFO_GLM53_LAYERS=8;GUFO_GLM53_LOGITS=1"
    COST 3600)
endif()

# Independent numpy reimplementation cross-checks every dumped stage.
if(PYTHON3_EXECUTABLE)
  add_test(NAME glm53.layer0_numpy
    COMMAND ${PYTHON3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/layer0_numpy.py
      "${GUFO_GLM53_FLASH_MODEL}"
      ${CMAKE_CURRENT_BINARY_DIR}/glm53-dump)
  set_tests_properties(glm53.layer0_numpy PROPERTIES
    LABELS "cpu;glm53;reference;external-model"
    SKIP_RETURN_CODE 77
    FIXTURES_REQUIRED glm53-dump
    ENVIRONMENT "GUFO_GLM53_LAYERS=4"
    COST 300)
endif()

add_executable(glm53_reference_check reference_check.cpp)
target_link_libraries(glm53_reference_check PRIVATE gufo_glm53_flash)
add_test(NAME glm53.reference_full
  COMMAND glm53_reference_check "${GUFO_GLM53_FLASH_MODEL}")
set_tests_properties(glm53.reference_full PROPERTIES
  LABELS "cpu;glm53;reference;external-model;slow"
  SKIP_RETURN_CODE 77
  ENVIRONMENT "GUFO_GLM53_GOLDENS=$ENV{GUFO_GLM53_GOLDENS}"
  COST 3600)

if(ENGINE_ENABLE_HIP)
  add_executable(glm53_matvec_test matvec_test.cpp)
  target_link_libraries(glm53_matvec_test PRIVATE gufo_glm53_flash_rocm)
  add_test(NAME glm53.matvec
    COMMAND glm53_matvec_test "${GUFO_GLM53_FLASH_MODEL}")
  set_tests_properties(glm53.matvec PROPERTIES
    LABELS "gpu;glm53;kernel;external-model"
    SKIP_RETURN_CODE 77)

  add_executable(glm53_mhc_test mhc_test.cpp)
  target_link_libraries(glm53_mhc_test PRIVATE gufo_glm53_flash_rocm)
  add_test(NAME glm53.mhc
    COMMAND glm53_mhc_test "${GUFO_GLM53_FLASH_MODEL}")
  set_tests_properties(glm53.mhc PROPERTIES
    LABELS "gpu;glm53;kernel;external-model"
    SKIP_RETURN_CODE 77)

  add_executable(glm53_kda_test kda_test.cpp)
  target_link_libraries(glm53_kda_test PRIVATE gufo_glm53_flash_rocm)
  add_test(NAME glm53.kda
    COMMAND glm53_kda_test "${GUFO_GLM53_FLASH_MODEL}")
  set_tests_properties(glm53.kda PROPERTIES
    LABELS "gpu;glm53;kernel;external-model"
    SKIP_RETURN_CODE 77)

  add_executable(glm53_attend_test attend_test.cpp)
  target_link_libraries(glm53_attend_test PRIVATE gufo_glm53_flash_rocm)

  add_executable(glm53_dense_bench EXCLUDE_FROM_ALL dense_bench.cpp)
  target_link_libraries(glm53_dense_bench PRIVATE gufo_glm53_flash_engine)

  add_executable(glm53_hc_mix_test hc_mix_test.cpp)
  target_link_libraries(glm53_hc_mix_test PRIVATE gufo_glm53_flash_rocm)

  add_executable(glm53_dsa_test dsa_test.cpp)
  target_link_libraries(glm53_dsa_test PRIVATE gufo_glm53_flash_rocm)
  add_test(NAME glm53.dsa
    COMMAND glm53_dsa_test "${GUFO_GLM53_FLASH_MODEL}")
  set_tests_properties(glm53.dsa PROPERTIES
    LABELS "gpu;glm53;kernel;external-model"
    SKIP_RETURN_CODE 77
    COST 120)

  add_executable(glm53_moe_test moe_test.cpp)
  target_link_libraries(glm53_moe_test PRIVATE gufo_glm53_flash_rocm)
  add_test(NAME glm53.moe
    COMMAND glm53_moe_test "${GUFO_GLM53_FLASH_MODEL}")
  set_tests_properties(glm53.moe PROPERTIES
    LABELS "gpu;glm53;kernel;external-model"
    SKIP_RETURN_CODE 77
    COST 120)

  add_executable(glm53_executor_test executor_test.cpp)
  target_link_libraries(glm53_executor_test PRIVATE
    gufo_glm53_flash_engine gufo_glm53_flash)
  add_test(NAME glm53.executor
    COMMAND glm53_executor_test "${GUFO_GLM53_FLASH_MODEL}")
  set_tests_properties(glm53.executor PROPERTIES
    LABELS "gpu;glm53;reference;external-model;slow"
    SKIP_RETURN_CODE 77
    COST 600)

  add_executable(glm53_executor_full_test executor_full_test.cpp)
  target_link_libraries(glm53_executor_full_test PRIVATE
    gufo_glm53_flash_engine gufo_glm53_flash)
  add_test(NAME glm53.executor_full
    COMMAND glm53_executor_full_test "${GUFO_GLM53_FLASH_MODEL}")
  set_tests_properties(glm53.executor_full PROPERTIES
    LABELS "gpu;glm53;reference;external-model;slow"
    SKIP_RETURN_CODE 77
    ENVIRONMENT "GUFO_GLM53_GOLDENS=$ENV{GUFO_GLM53_GOLDENS}"
    COST 1200)

  add_executable(glm53_executor_long_test executor_long_test.cpp)
  target_link_libraries(glm53_executor_long_test PRIVATE
    gufo_glm53_flash_engine gufo_glm53_flash)

  add_executable(glm53_gemm_test gemm_test.cpp)
  target_link_libraries(glm53_gemm_test PRIVATE gufo_glm53_flash_rocm)
  add_test(NAME glm53.gemm
    COMMAND glm53_gemm_test "${GUFO_GLM53_FLASH_MODEL}")
  set_tests_properties(glm53.gemm PROPERTIES
    LABELS "gpu;glm53;kernel;external-model"
    SKIP_RETURN_CODE 77
    COST 120)

  add_executable(glm53_executor_prefill_test executor_prefill_test.cpp)
  target_link_libraries(glm53_executor_prefill_test PRIVATE
    gufo_glm53_flash_engine gufo_glm53_flash)

  add_executable(glm53_experts_bench EXCLUDE_FROM_ALL experts_bench.cpp)
  target_link_libraries(glm53_experts_bench PRIVATE gufo_glm53_flash_rocm)

  add_executable(glm53_routed_f16_test routed_f16_test.cpp)
  target_link_libraries(glm53_routed_f16_test PRIVATE gufo_qwen38_flash_next
    gufo_core)
  add_test(NAME glm53.routed_f16
    COMMAND glm53_routed_f16_test "${GUFO_GLM53_FLASH_MODEL}")
  set_tests_properties(glm53.routed_f16 PROPERTIES
    LABELS "gpu;glm53;kernel;external-model"
    SKIP_RETURN_CODE 77
    COST 120)

  add_executable(glm53_spec_bench EXCLUDE_FROM_ALL spec_bench.cpp)
  target_link_libraries(glm53_spec_bench PRIVATE gufo_glm53_flash_engine
    gufo_glm53_flash)
endif()
