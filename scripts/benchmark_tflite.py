"""
Phase 12 — Benchmark TFLite model on host

This is a host-side smoke benchmark. Real latency must be measured on
Grove Vision AI V2 with Ethos-U55 enabled.

Usage:
    python scripts/benchmark_tflite.py models/tflite/best_int8.tflite --runs 100
"""

import argparse
import time
from pathlib import Path

import numpy as np


def benchmark(model_path: str, runs: int = 100, warmup: int = 10):
    import tensorflow as tf

    model_path = Path(model_path)
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    shape = inp["shape"]
    dtype = inp["dtype"]

    if np.issubdtype(dtype, np.integer):
        data = np.random.randint(-128, 127, size=shape, dtype=dtype)
    else:
        data = np.random.random_sample(shape).astype(dtype)

    for _ in range(warmup):
        interpreter.set_tensor(inp["index"], data)
        interpreter.invoke()

    times = []
    for _ in range(runs):
        interpreter.set_tensor(inp["index"], data)
        start = time.perf_counter()
        interpreter.invoke()
        times.append((time.perf_counter() - start) * 1000.0)

    times = np.array(times)
    print(f"Model: {model_path}")
    print(f"Runs: {runs}")
    print(f"Mean latency: {times.mean():.2f} ms")
    print(f"P50 latency:  {np.percentile(times, 50):.2f} ms")
    print(f"P95 latency:  {np.percentile(times, 95):.2f} ms")
    print(f"Host FPS:     {1000.0 / times.mean():.2f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark a TFLite model")
    parser.add_argument("model")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    benchmark(args.model, args.runs, args.warmup)


if __name__ == "__main__":
    main()
