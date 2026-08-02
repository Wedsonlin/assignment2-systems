from __future__ import annotations

import argparse
import gc
import json
from itertools import product

import torch
from triton.testing import do_bench

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.flash_attention import FlashAttentionV2Triton

SEQUENCE_LENGTHS = tuple(2**power for power in range(7, 17))
EMBEDDING_DIMS = (16, 32, 64, 128)
BACKENDS = ("pytorch", "triton")
MODES = ("forward", "backward", "forward_backward")


def _release_cuda_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PyTorch attention against Triton FlashAttention-2.")
    parser.add_argument("--seq-lens", type=lambda value: [int(item) for item in value.split(",")], default=list(SEQUENCE_LENGTHS))
    parser.add_argument("--dims", type=lambda value: [int(item) for item in value.split(",")], default=list(EMBEDDING_DIMS))
    parser.add_argument("--dtypes", type=lambda value: value.lower().split(","), default=["bf16", "fp32"])
    parser.add_argument("--backends", type=lambda value: value.lower().split(","), default=list(BACKENDS))
    parser.add_argument("--modes", type=lambda value: value.lower().split(","), default=list(MODES))
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    args = parser.parse_args()

    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_device(args.device)

    configurations = product(args.seq_lens, args.dims, args.dtypes)
    for sequence_length, embedding_dim, dtype_name in configurations:
        dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
        q = torch.randn(1, sequence_length, embedding_dim, device=args.device, dtype=dtype, requires_grad=True)
        k = torch.randn(1, sequence_length, embedding_dim, device=args.device, dtype=dtype, requires_grad=True)
        v = torch.randn(1, sequence_length, embedding_dim, device=args.device, dtype=dtype, requires_grad=True)
        d_o = torch.randn_like(q)

        for backend in args.backends:
            causal_mask = None
            if backend == "pytorch":
                def forward() -> torch.Tensor:
                    return scaled_dot_product_attention(q, k, v, mask=causal_mask)
            else:
                def forward() -> torch.Tensor:
                    return FlashAttentionV2Triton.apply(q, k, v, True)

            for mode in args.modes:
                measured = None
                output = None
                try:
                    if mode == "forward":
                        measured = forward
                    elif mode == "backward":
                        output = forward()
                        def measured(output: torch.Tensor = output) -> None:
                            output.backward(d_o, retain_graph=True)
                    else:
                        def measured() -> None:
                            forward().backward(d_o)
                    mean_ms = do_bench(
                                measured,
                                warmup=args.warmup_ms,
                                rep=args.rep_ms,
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

                print(
                    json.dumps(
                        {
                            "seq_len": sequence_length,
                            "d_model": embedding_dim,
                            "precision": dtype_name,
                            "mode": mode,
                            "backend": backend,
                            "mean_ms": mean_ms,
                        }
                    )
                )

            forward = None
            causal_mask = None
            _release_cuda_memory()

        del q, k, v, d_o
        _release_cuda_memory()
