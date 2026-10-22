# GLM-5.3-Flash CTest checks. Artifact-backed tests skip (77) without
# GUFO_GLM53_FLASH_MODEL so hosted CI stays green.
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
