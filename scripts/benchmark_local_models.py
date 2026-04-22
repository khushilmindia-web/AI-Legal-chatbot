from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests


def load_eval_prompts(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        items.append(payload)
        if limit is not None and len(items) >= limit:
            break
    return items


def benchmark_ollama(*, model: str, prompts: list[dict[str, Any]], host: str, timeout: float) -> dict[str, Any]:
    timings: list[float] = []
    lengths: list[int] = []
    for item in prompts:
        start = time.perf_counter()
        response = requests.post(
            f"{host.rstrip('/')}/api/generate",
            json={"model": model, "prompt": item["prompt"], "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        text = str(payload.get("response") or "")
        timings.append(time.perf_counter() - start)
        lengths.append(len(text))
    return summarize_results(engine="ollama", model=model, timings=timings, lengths=lengths, sample_count=len(prompts))


def benchmark_openai_compatible(*, model: str, prompts: list[dict[str, Any]], base_url: str, timeout: float) -> dict[str, Any]:
    timings: list[float] = []
    lengths: list[int] = []
    for item in prompts:
        start = time.perf_counter()
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": item["prompt"]}],
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        text = str(payload.get("choices", [{}])[0].get("message", {}).get("content") or "")
        timings.append(time.perf_counter() - start)
        lengths.append(len(text))
    return summarize_results(engine="openai_compatible", model=model, timings=timings, lengths=lengths, sample_count=len(prompts))


def summarize_results(*, engine: str, model: str, timings: list[float], lengths: list[int], sample_count: int) -> dict[str, Any]:
    return {
        "engine": engine,
        "model": model,
        "sample_count": sample_count,
        "latency_seconds": {
            "mean": round(statistics.mean(timings), 4) if timings else 0.0,
            "median": round(statistics.median(timings), 4) if timings else 0.0,
            "max": round(max(timings), 4) if timings else 0.0,
        },
        "response_chars": {
            "mean": round(statistics.mean(lengths), 1) if lengths else 0.0,
            "median": round(statistics.median(lengths), 1) if lengths else 0.0,
            "max": max(lengths) if lengths else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local generation backends against the compact legal-bot eval set.")
    parser.add_argument("--engine", choices=["ollama", "openai_compatible"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-set", default="data/evals/legal_bot_eval_set.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", default="data/evals/local_model_benchmark_report.json")
    args = parser.parse_args()

    prompts = load_eval_prompts(Path(args.eval_set), limit=args.limit)
    if args.engine == "ollama":
        report = benchmark_ollama(model=args.model, prompts=prompts, host=args.host, timeout=args.timeout)
    else:
        report = benchmark_openai_compatible(model=args.model, prompts=prompts, base_url=args.base_url, timeout=args.timeout)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
