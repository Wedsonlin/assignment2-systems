import argparse
import gc
import json
import os
import timeit
from pathlib import Path
from typing import Type

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import Optimizer

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.ddp import OptimizedDDP


class ShardedOptimizer(Optimizer):
    @torch.no_grad()
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs):
        self.list_params = []
        ids = set()
        for p in params:
            if id(p) not in ids and p.requires_grad:
                self.list_params.append(p)
                ids.add(id(p))

        n_params = sum(p.numel() for p in self.list_params)
        self.flatten_buffer = torch.empty(n_params, dtype=self.list_params[0].dtype, device=self.list_params[0].device)

        offest = 0
        for p in self.list_params:
            self.flatten_buffer[offest:offest+p.numel()].copy_(p.data.view(-1))
            p.set_(self.flatten_buffer[offest:offest+p.numel()].view_as(p))
            offest += p.numel()

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        offset_per_rank = n_params // world_size # rquire n_params % world_size == 0
        self.start = rank * offset_per_rank
        self.end = self.start + offset_per_rank
        self.shard = self.flatten_buffer[self.start:self.end]

        self.local_params = []
        self.local_mappings = []
        offset = 0
        for p in self.list_params:
            global_start = max(offset, self.start)
            global_end  = min(offset + p.numel(), self.end)
            if global_start < global_end:
                local_param = self.flatten_buffer[global_start:global_end]
                local_start = global_start - offset
                local_end = global_end - offset

                self.local_params.append(local_param)
                self.local_mappings.append(
                    (local_param, p, local_start, local_end)
                )
            offset += p.numel()

        super().__init__(self.list_params, defaults={})
        self.optimizer = optimizer_cls(self.local_params, **kwargs)

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none)
        self.optimizer.zero_grad(set_to_none=True)

    def step(self, closure=None, **kwargs):
        for local_param, p, start, end in self.local_mappings:
            local_param.grad = p.grad.view(-1)[start:end]

        self.optimizer.step(closure, **kwargs)
        dist.all_gather_into_tensor(self.flatten_buffer, self.shard)


GIB = 1024**3


def memory_snapshot(model, optimizer=None):
    torch.cuda.synchronize()
    parameters = sum(p.numel() * p.element_size() for p in model.parameters())
    gradients = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
    inner_optimizer = optimizer.optimizer if isinstance(optimizer, ShardedOptimizer) else optimizer
    optimizer_states = 0 if inner_optimizer is None else sum(
        value.numel() * value.element_size()
        for state in inner_optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )
    allocated = torch.cuda.memory_allocated()
    result = {
        "allocated_gib": allocated / GIB,
        "phase_peak_gib": torch.cuda.max_memory_allocated() / GIB,
        "parameters_gib": parameters / GIB,
        "gradients_gib": gradients / GIB,
        "optimizer_states_gib": optimizer_states / GIB,
        "other_gib": (allocated - parameters - gradients - optimizer_states) / GIB,
    }
    torch.cuda.reset_peak_memory_stats()
    return {key: round(value, 3) for key, value in result.items()}


def compute_loss(model, inputs, targets, mixed_precision: bool):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision):
        return cross_entropy(model(inputs), targets)


def train_step(model, optimizer, inputs, targets, mixed_precision: bool):
    optimizer.zero_grad(set_to_none=True)
    loss = compute_loss(model, inputs, targets, mixed_precision)
    loss.backward()
    model.finish_gradient_synchronization()
    optimizer.step()


def run_experiment(rank, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(args.master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    device = torch.device("cuda", rank)
    local_batch_size = args.batch_size // args.world_size
    modes = [False, True] if args.mode == "both" else [args.mode == "sharded"]
    results = []

    for sharded in modes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        ).to(device)
        model = OptimizedDDP(model)
        memory = {"after_model_init": memory_snapshot(model)}

        optimizer_cls = torch.optim.AdamW
        optimizer = (
            ShardedOptimizer(model.parameters(), optimizer_cls, lr=args.lr, foreach=False)
            if sharded
            else optimizer_cls(model.parameters(), lr=args.lr, foreach=False)
        )
        inputs = torch.randint(args.vocab_size, (local_batch_size, args.context_length), device=device)
        targets = torch.randint(args.vocab_size, (local_batch_size, args.context_length), device=device)

        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, inputs, targets, args.mixed_precision)
        loss.backward()
        model.finish_gradient_synchronization()
        memory["before_optimizer_step"] = memory_snapshot(model, optimizer)
        optimizer.step()
        memory["after_optimizer_step"] = memory_snapshot(model, optimizer)
        del loss

        for _ in range(args.warmup_steps):
            train_step(model, optimizer, inputs, targets, args.mixed_precision)
        dist.barrier()
        torch.cuda.synchronize()
        start = timeit.default_timer()
        for _ in range(args.steps):
            train_step(model, optimizer, inputs, targets, args.mixed_precision)
        torch.cuda.synchronize()
        iteration_ms = torch.tensor((timeit.default_timer() - start) * 1000 / args.steps, device=device)
        dist.all_reduce(iteration_ms, op=dist.ReduceOp.MAX)

        memory_per_rank = [None] * args.world_size if rank == 0 else None
        dist.gather_object({"rank": rank, "phases": memory}, memory_per_rank, dst=0)

        if rank == 0:
            max_phase_peaks = {}
            for phase in memory:
                worst = max(
                    memory_per_rank,
                    key=lambda item: item["phases"][phase]["phase_peak_gib"],
                )
                max_phase_peaks[phase] = {
                    "rank": worst["rank"],
                    "phase_peak_gib": worst["phases"][phase]["phase_peak_gib"],
                }
            run_peak_phase = max(
                max_phase_peaks,
                key=lambda phase: max_phase_peaks[phase]["phase_peak_gib"],
            )
            results.append(
                {
                    "mode": "sharded" if sharded else "baseline",
                    "mixed_precision": args.mixed_precision,
                    "compute_dtype": "bf16" if args.mixed_precision else "fp32",
                    "num_parameters": sum(p.numel() for p in model.parameters()),
                    "memory": {
                        "per_rank": memory_per_rank,
                        "max_phase_peaks": max_phase_peaks,
                        "run_peak": {
                            "phase": run_peak_phase,
                            **max_phase_peaks[run_peak_phase],
                        },
                    },
                    "iteration_ms": round(iteration_ms.item(), 3),
                }
            )

        del inputs, targets, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0:
        output = json.dumps({"config": vars(args), "results": results}, indent=2)
        print(output)
        output_path = Path(__file__).resolve().parents[1] / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Profile optimizer-state sharding memory and iteration time.")
    parser.add_argument("--mode", choices=("both", "baseline", "sharded"), default="both")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=2560)
    parser.add_argument("--d-ff", type=int, default=10240)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--master-port", type=int, default=12391)
    parser.add_argument("--output", default="results/sharded_optimizer.json")
    args = parser.parse_args()
    mp.spawn(run_experiment, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()

