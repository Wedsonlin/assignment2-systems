import argparse
import gc
import json
import os
import timeit
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import Optimizer

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.ddp import OptimizedDDP
from cs336_systems.profile_utils import memory_snapshot, summarize_memory


class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs):
        self.params = [param for param in params if param.requires_grad]
        super().__init__(self.params, defaults={})
        rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # ponytail: parameter-level sharding; flatten only if measured imbalance matters.
        self.optimizer = optimizer_cls(self.params[rank::self.world_size], **kwargs)

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)

    def step(self, closure=None, **kwargs):
        loss = self.optimizer.step(closure, **kwargs)
        with torch.no_grad():
            for index, param in enumerate(self.params):
                dist.broadcast(param, src=index % self.world_size)
        return loss


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
            results.append(
                {
                    "mode": "sharded" if sharded else "baseline",
                    "mixed_precision": args.mixed_precision,
                    "compute_dtype": "bf16" if args.mixed_precision else "fp32",
                    "num_parameters": sum(p.numel() for p in model.parameters()),
                    "memory": summarize_memory(memory, memory_per_rank),
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
