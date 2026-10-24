#!/usr/bin/env python3
"""HumanEval pass@1 against an OpenAI-compatible chat server.

Greedy by default. Each task is sent as a chat request asking for the complete
function; the first fenced Python block (or the whole reply) is executed with
the task's own tests in a fresh interpreter. The dataset is the 164-problem
openai/human-eval JSONL, passed with --dataset.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def chat(base_url: str, model: str, prompt: str, max_tokens: int,
         effort: str | None, timeout: float) -> dict:
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": "Complete the following Python function. Reply with "
                       "the complete function (including its signature) in "
                       "one ```python block and nothing else.\n\n" + prompt,
        }],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if effort is not None:
        body["reasoning_effort"] = effort
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def extract_code(text: str, prompt: str, entry_point: str) -> str:
    blocks = FENCE.findall(text)
    code = max(blocks, key=len) if blocks else text
    if f"def {entry_point}" in code:
        # The reply restates the function; keep the prompt's imports and
        # helper definitions ahead of it.
        header = prompt.split(f"def {entry_point}")[0]
        return header + "\n" + code
    return prompt + code


def run_tests(program: str, test: str, entry_point: str,
              timeout: float) -> tuple[bool, str]:
    source = f"{program}\n\n{test}\n\ncheck({entry_point})\n"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "task.py"
        path.write_text(source, encoding="utf-8")
        try:
            run = subprocess.run([sys.executable, "-I", str(path)],
                                 capture_output=True, text=True,
                                 timeout=timeout, cwd=directory)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if run.returncode == 0:
        return True, ""
    return False, (run.stderr.strip().splitlines() or ["failed"])[-1][:200]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="model")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--effort", default="none",
                        help="reasoning_effort sent with each request; "
                             "'none' disables reasoning")
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--test-timeout", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tasks = [json.loads(line) for line in
             args.dataset.read_text(encoding="utf-8").splitlines() if line]
    if args.limit:
        tasks = tasks[:args.limit]
    effort = None if args.effort == "default" else args.effort

    results: list[dict] = []
    started = time.time()
    tests = ThreadPoolExecutor(max_workers=args.workers)
    pending = []
    for index, task in enumerate(tasks):
        began = time.time()
        reply = chat(args.base_url, args.model, task["prompt"],
                     args.max_tokens, effort, args.request_timeout)
        message = reply["choices"][0]["message"]
        text = message.get("content") or ""
        usage = reply.get("usage", {})
        program = extract_code(text, task["prompt"], task["entry_point"])
        future = tests.submit(run_tests, program, task["test"],
                              task["entry_point"], args.test_timeout)
        pending.append((task, text, usage, future, time.time() - began))
        done = sum(1 for p in pending if p[3].done() and p[3].result()[0])
        print(f"[{index + 1}/{len(tasks)}] {task['task_id']} "
              f"{usage.get('completion_tokens', 0)} tok "
              f"{time.time() - began:.1f}s (passed so far {done})",
              file=sys.stderr, flush=True)
    passed = 0
    for task, text, usage, future, seconds in pending:
        ok, error = future.result()
        passed += ok
        results.append({
            "task_id": task["task_id"], "passed": ok, "error": error,
            "completion_tokens": usage.get("completion_tokens", 0),
            "seconds": seconds, "completion": text})
    tests.shutdown()
    summary = {
        "benchmark": "HumanEval", "tasks": len(results), "passed": passed,
        "pass_at_1": passed / max(1, len(results)),
        "effort": args.effort, "max_tokens": args.max_tokens,
        "wall_seconds": time.time() - started,
        "completion_tokens": sum(r["completion_tokens"] for r in results),
    }
    print(json.dumps(summary, indent=1))
    if args.output:
        args.output.write_text(
            json.dumps({"summary": summary, "results": results}, indent=1),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
