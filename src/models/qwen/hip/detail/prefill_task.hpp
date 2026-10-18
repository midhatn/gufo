#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_PREFILL_TASK_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_PREFILL_TASK_HPP_

#include <coroutine>
#include <exception>
#include <utility>

#include "src/models/qwen/tokenizer.hpp"

namespace gufo::hip::detail {

/// One prefill chunk as a resumable task. The body suspends right after it
/// queues an NPU stage, so a driver can run another chunk's GPU work while the
/// NPU computes. Run alone it is resumed until done.
class PrefillTask {
public:
  struct promise_type {
    tokenization::TokenId value{0};
    std::exception_ptr error;

    PrefillTask get_return_object() {
      return PrefillTask(
          std::coroutine_handle<promise_type>::from_promise(*this));
    }
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    std::suspend_always yield_value(int /*stage*/) noexcept { return {}; }
    void return_value(tokenization::TokenId token) noexcept { value = token; }
    void unhandled_exception() noexcept { error = std::current_exception(); }
  };

  PrefillTask(PrefillTask&& other) noexcept
      : handle_(std::exchange(other.handle_, {})) {}
  PrefillTask& operator=(PrefillTask&& other) noexcept {
    if (this != &other) {
      Destroy();
      handle_ = std::exchange(other.handle_, {});
    }
    return *this;
  }
  PrefillTask(const PrefillTask&) = delete;
  PrefillTask& operator=(const PrefillTask&) = delete;
  ~PrefillTask() { Destroy(); }

  [[nodiscard]] bool Done() const noexcept { return handle_.done(); }
  /// Runs to the next suspension point; rethrows an exception from the body.
  void Resume() {
    handle_.resume();
    if (handle_.promise().error) {
      std::rethrow_exception(std::exchange(handle_.promise().error, nullptr));
    }
  }
  [[nodiscard]] tokenization::TokenId Result() const {
    return handle_.promise().value;
  }

private:
  explicit PrefillTask(std::coroutine_handle<promise_type> handle)
      : handle_(handle) {}
  void Destroy() noexcept {
    if (handle_) {
      handle_.destroy();
      handle_ = {};
    }
  }

  std::coroutine_handle<promise_type> handle_;
};

}  // namespace gufo::hip::detail

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_PREFILL_TASK_HPP_
