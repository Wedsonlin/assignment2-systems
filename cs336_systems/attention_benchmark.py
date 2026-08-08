"""Benchmark attention for the three assignment suites.

1. pytorch_attention: vanilla PyTorch attention (batch=8, fwd/bwd + activation memory)
2. torch_compile: same config with torch.compile
3. flash: PyTorch vs Triton FlashAttention-2 (batch=1, causal, do_bench)
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Any

import torch
from triton.testing import do_bench

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.flash_attention import FlashAttentionV2Triton

VANILLA_SEQ_LENGTHS = (256, 1024, 4096, 8192, 16384)
VANILLA_EMBEDDING_DIMS = (16, 32, 64, 128)
FLASH_SEQ_LENGTHS = tuple(2**power for power in range(7, 17))
FLASH_EMBEDDING_DIMS = (16, 32, 64, 128)
FLASH_MODES = ("forward", "backward", "forward_backward")
SUITES = ("pytorch_attention", "torch_compile", "flash")
DEFAULT_OUTPUT = Path("results/attention_benchmark.jsonl")


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_csv_strs(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _release_cuda_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _make_causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    iota = torch.arange(sequence_length, device=device)
    return iota[:, None] >= iota[None, :]


def _bytes_to_mib(num_bytes: int) -> float:
    return round(num_bytes / (1024**2), 3)


def _write_record(output_file, record: dict[str, Any]) -> None:
    line = json.dumps(record)
    print(line)
    output_file.write(line + "\n")
    output_file.flush()


def _time_cuda_ms(fn: Callable[[], Any], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.cuda.synchronize()
    end = time.perf_counter()
    return round((end - start) * 1000.0 / iters, 3)


def _make_qkv(
    batch_size: int,
    sequence_length: int,
    embedding_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch_size, sequence_length, embedding_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, sequence_length, embedding_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, sequence_length, embedding_dim, device=device, dtype=dtype, requires_grad=True)
    d_o = torch.randn_like(q)
    return q, k, v, d_o


def _make_forward(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    compiled_attn: Callable[..., torch.Tensor] | None = None,
) -> Callable[[], torch.Tensor]:
    mask = _make_causal_mask(q.shape[1], q.device) if causal else None

    if backend == "pytorch":
        def forward() -> torch.Tensor:
            return scaled_dot_product_attention(q, k, v, mask=mask)

        return forward

    if backend == "compile":
        assert compiled_attn is not None

        def forward() -> torch.Tensor:
            return compiled_attn(q, k, v, mask=mask)

        return forward

    if backend == "triton":
        def forward() -> torch.Tensor:
            return FlashAttentionV2Triton.apply(q, k, v, causal)

        return forward

    raise ValueError(f"Unsupported backend: {backend}")


def _run_vanilla_suite(
    output_file,
    *,
    suite: str,
    seq_lens: list[int],
    dims: list[int],
    dtypes: list[str],
    batch_size: int,
    causal: bool,
    warmup: int,
    iters: int,
    device: torch.device,
) -> None:
    backend = "compile" if suite == "torch_compile" else "pytorch"
    compiled_attn = torch.compile(scaled_dot_product_attention) if backend == "compile" else None
    # torch.compile + retain_graph=True conflicts with donated buffers.
    if backend == "compile":
        torch._functorch.config.donated_buffer = False

    for sequence_length, embedding_dim, dtype_name in product(seq_lens, dims, dtypes):
        dtype = _dtype_from_name(dtype_name)
        q, k, v, d_o = _make_qkv(batch_size, sequence_length, embedding_dim, dtype, device)
        forward = _make_forward(backend, q, k, v, causal=causal, compiled_attn=compiled_attn)

        forward_ms: float | str = "OOM"
        backward_ms: float | str = "OOM"
        activation_mib: float | str = "OOM"

        try:
            # Warm compile / kernels before timed region.
            for _ in range(max(warmup, 1)):
                out = forward()
                out.backward(d_o)
                for tensor in (q, k, v):
                    tensor.grad = None
            torch.cuda.synchronize()
            _release_cuda_memory()

            forward_ms = _time_cuda_ms(forward, warmup=warmup, iters=iters)

            for tensor in (q, k, v):
                tensor.grad = None
            _release_cuda_memory()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

            output = forward()
            torch.cuda.synchronize()
            activation_mib = _bytes_to_mib(torch.cuda.memory_allocated(device))

            def backward() -> None:
                output.backward(d_o, retain_graph=True)

            backward_ms = _time_cuda_ms(backward, warmup=warmup, iters=iters)
        except torch.cuda.OutOfMemoryError:
            pass
        finally:
            for tensor in (q, k, v):
                tensor.grad = None
            forward = None
            _release_cuda_memory()

        record = {
            "suite": suite,
            "backend": backend,
            "seq_len": sequence_length,
            "d_model": embedding_dim,
            "precision": dtype_name,
            "batch_size": batch_size,
            "causal": causal,
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "activation_mib": activation_mib,
        }
        _write_record(output_file, record)
        del q, k, v, d_o
        _release_cuda_memory()


def _run_flash_suite(
    output_file,
    *,
    seq_lens: list[int],
    dims: list[int],
    dtypes: list[str],
    backends: list[str],
    modes: list[str],
    batch_size: int,
    causal: bool,
    warmup_ms: int,
    rep_ms: int,
    device: torch.device,
) -> None:
    for sequence_length, embedding_dim, dtype_name in product(seq_lens, dims, dtypes):
        dtype = _dtype_from_name(dtype_name)
        q, k, v, d_o = _make_qkv(batch_size, sequence_length, embedding_dim, dtype, device)

        for backend in backends:
            if backend == "compile":
                compiled_attn = torch.compile(scaled_dot_product_attention)
            else:
                compiled_attn = None
            forward = _make_forward(backend, q, k, v, causal=causal, compiled_attn=compiled_attn)

            for mode in modes:
                measured = None
                output = None
                mean_ms: float | str
                try:
                    if mode == "forward":
                        measured = forward
                    elif mode == "backward":
                        output = forward()

                        def measured(output: torch.Tensor = output) -> None:
                            output.backward(d_o, retain_graph=True)
                    elif mode == "forward_backward":
                        def measured() -> None:
                            forward().backward(d_o)
                    else:
                        raise ValueError(f"Unsupported mode: {mode}")

                    mean_ms = do_bench(
                        measured,
                        warmup=warmup_ms,
                        rep=rep_ms,
                        grad_to_none=[q, k, v],
                        return_mode="mean",
                    )
                    mean_ms = round(mean_ms, 3)
                except torch.cuda.OutOfMemoryError:
                    mean_ms = "OOM"
                finally:
                    for tensor in (q, k, v):
                        tensor.grad = None
                    measured = None
                    output = None
                    _release_cuda_memory()

                record = {
                    "suite": "flash",
                    "backend": backend,
                    "seq_len": sequence_length,
                    "d_model": embedding_dim,
                    "precision": dtype_name,
                    "batch_size": batch_size,
                    "causal": causal,
                    "mode": mode,
                    "mean_ms": mean_ms,
                }
                _write_record(output_file, record)

            forward = None
            compiled_attn = None
            _release_cuda_memory()

        del q, k, v, d_o
        _release_cuda_memory()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark attention for pytorch_attention, torch_compile, and flash suites. "
            "Use --suite to select one or more."
        )
    )
    parser.add_argument(
        "--suite",
        type=_parse_csv_strs,
        default=["flash"],
        help="Comma-separated: pytorch_attention,torch_compile,flash,all",
    )
    parser.add_argument("--seq-lens", type=_parse_csv_ints, default=None)
    parser.add_argument("--dims", type=_parse_csv_ints, default=None)
    parser.add_argument("--dtypes", type=_parse_csv_strs, default=None)
    parser.add_argument("--backends", type=_parse_csv_strs, default=None, help="Flash suite only: pytorch,compile,triton")
    parser.add_argument("--modes", type=_parse_csv_strs, default=None, help="Flash suite only: forward,backward,forward_backward")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--causal", type=lambda value: value.lower() in {"1", "true", "yes"}, default=None)
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iters for vanilla/compile suites")
    parser.add_argument("--iters", type=int, default=100, help="Timed iters for vanilla/compile suites")
    parser.add_argument("--warmup-ms", type=int, default=2000, help="do_bench warmup ms for flash suite (large enough to absorb Triton autotune)")
    parser.add_argument("--rep-ms", type=int, default=5000, help="do_bench rep ms for flash suite")
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    requested = set(args.suite)
    if "all" in requested:
        suites = list(SUITES)
    else:
        suites = [suite for suite in SUITES if suite in requested]
        unknown = requested - set(SUITES) - {"all"}
        if unknown:
            raise SystemExit(f"Unknown suite(s): {sorted(unknown)}")
        if not suites:
            raise SystemExit("No valid --suite selected")

    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output_file:
        for suite in suites:
            if suite in {"pytorch_attention", "torch_compile"}:
                _run_vanilla_suite(
                    output_file,
                    suite=suite,
                    seq_lens=args.seq_lens or list(VANILLA_SEQ_LENGTHS),
                    dims=args.dims or list(VANILLA_EMBEDDING_DIMS),
                    dtypes=args.dtypes or ["fp32"],
                    batch_size=8 if args.batch_size is None else args.batch_size,
                    causal=False if args.causal is None else args.causal,
                    warmup=args.warmup,
                    iters=args.iters,
                    device=args.device,
                )
            else:
                backends = args.backends or ["pytorch", "triton"]
                modes = args.modes or list(FLASH_MODES)
                _run_flash_suite(
                    output_file,
                    seq_lens=args.seq_lens or list(FLASH_SEQ_LENGTHS),
                    dims=args.dims or list(FLASH_EMBEDDING_DIMS),
                    dtypes=args.dtypes or ["bf16", "fp32"],
                    backends=backends,
                    modes=modes,
                    batch_size=1 if args.batch_size is None else args.batch_size,
                    causal=True if args.causal is None else args.causal,
                    warmup_ms=args.warmup_ms,
                    rep_ms=args.rep_ms,
                    device=args.device,
                )


if __name__ == "__main__":
    main()
