"""
benchmark.py — Measures GPT-2 decode throughput (tokens/sec) on different devices.

Usage:
    python benchmark.py --checkpoint path/to/checkpoint.pt [--max_tokens 200] [--runs 3]
"""

import argparse
import time
import torch
import torch.nn.functional as F
import statistics

from base.model.gpt2 import GPT2
from gpt2_infer import load_gpt2_checkpoint, TiktokenWrapper


PROMPT = "Once upon a time in a land far away"
WARMUP_TOKENS = 20   # tokens generated during warmup (discarded)


def get_available_devices():
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")
    if torch.cuda.is_available():
        devices.insert(0, "cuda")
    return devices


@torch.no_grad()
def benchmark_device(checkpoint_path, device, max_tokens=200, runs=3):
    print(f"\n{'='*60}")
    print(f"  Benchmarking on: {device.upper()}")
    print(f"{'='*60}")

    model, config = load_gpt2_checkpoint(checkpoint_path, device)
    tokenizer = TiktokenWrapper()

    token_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(device)

    # ── Warmup ──────────────────────────────────────────────────
    print(f"  Warming up ({WARMUP_TOKENS} tokens)...", end=" ", flush=True)
    count = 0
    for _ in model.generate(token_ids, tokenizer, max_new_tokens=WARMUP_TOKENS,
                             temperature=0.8, top_k=50):
        count += 1
    # Sync device after warmup
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    print("done")

    # ── Timed runs ───────────────────────────────────────────────
    tok_per_sec_list = []
    for run in range(1, runs + 1):
        token_ids_run = tokenizer.encode(PROMPT, return_tensors="pt").to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()
        n_generated = 0
        for _ in model.generate(token_ids_run, tokenizer, max_new_tokens=max_tokens,
                                 temperature=0.8, top_k=50):
            n_generated += 1

        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

        elapsed = time.perf_counter() - t0
        tps = n_generated / elapsed
        tok_per_sec_list.append(tps)
        print(f"  Run {run}/{runs}: {n_generated} tokens in {elapsed:.2f}s → {tps:.1f} tok/s")

    mean_tps = statistics.mean(tok_per_sec_list)
    stdev_tps = statistics.stdev(tok_per_sec_list) if runs > 1 else 0.0
    print(f"\n{device.upper()} average: {mean_tps:.1f} ± {stdev_tps:.1f} tok/s")
    return mean_tps, stdev_tps


def main():
    parser = argparse.ArgumentParser(description="GPT-2 token/s benchmark")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to GPT-2 checkpoint (.pt)")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Number of tokens to generate per run")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of timed runs per device")
    parser.add_argument("--devices", type=str, default=None,
                        help="Comma-separated list of devices to test (e.g. mps,cpu). "
                             "Defaults to all available devices.")
    args = parser.parse_args()

    if args.devices:
        devices = [d.strip() for d in args.devices.split(",")]
    else:
        devices = get_available_devices()

    print(f"\nGPT-2 Decode Throughput Benchmark")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Prompt     : \"{PROMPT}\"")
    print(f"  Max tokens : {args.max_tokens}")
    print(f"  Runs       : {args.runs}")
    print(f"  Devices    : {', '.join(devices)}")

    results = {}
    for device in devices:
        mean_tps, stdev_tps = benchmark_device(
            args.checkpoint, device,
            max_tokens=args.max_tokens,
            runs=args.runs
        )
        results[device] = (mean_tps, stdev_tps)

    print(f"\n{'='*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Device':<8}  {'Avg tok/s':>12}  {'Std dev':>10}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*10}")
    for dev, (mean, std) in results.items():
        print(f"  {dev.upper():<8}  {mean:>12.1f}  {std:>10.1f}")
    print()


if __name__ == "__main__":
    main()
