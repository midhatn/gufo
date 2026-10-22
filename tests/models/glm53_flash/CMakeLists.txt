# GLM-5.3-Flash CTest checks. Artifact-backed tests skip (77) without
# GUFO_GLM53_FLASH_MODEL so hosted CI stays green.
set(GUFO_GLM53_FLASH_MODEL "$ENV{GUFO_GLM53_FLASH_MODEL}" CACHE FILEPATH
  "GLM-5.3-Flash GGUF shard 1 for external-model tests")
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

# Independent numpy reimplementation cross-checks every dumped stage.
find_program(PYTHON3_EXECUTABLE python3)
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
