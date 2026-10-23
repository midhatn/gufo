# GLM-5.3-Flash (`glm5-next`). The CPU side owns metadata, the tensor manifest
# and the tokenizer-adjacent contracts; GPU kernels live under kernels/rocm and
# are added once the layer-0 reference matches.
add_library(gufo_glm53_flash STATIC
  runtime/config.cpp
  runtime/manifest.cpp
  runtime/layout.cpp
  chat_template.cpp
  reference.cpp)

target_include_directories(gufo_glm53_flash PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(gufo_glm53_flash PUBLIC gufo_core)

# Host-only artifact check: parses the pinned `glm5-next` metadata and rejects
# anything this build cannot run. Not part of the production package.
add_executable(gufo_glm53_probe EXCLUDE_FROM_ALL probe_main.cpp)
target_link_libraries(gufo_glm53_probe PRIVATE gufo_glm53_flash)
