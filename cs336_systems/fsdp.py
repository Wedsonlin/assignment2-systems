import argparse
import json
import os
import timeit
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM, Embedding, Linear
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.profile_utils import memory_snapshot, summarize_memory


class FSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype


        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        self._forward_order = None
        self._recorded_order = []
        self._forward_index = 0
        self._pending_gather = {}
        self._shards = {}
        self._grad_works = []
        
        hook_params = set()

        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0) # maybe each rank uses different RNG, has different weights
            for submodule in self.module.modules():
                is_sharded = isinstance(submodule, (Embedding, Linear))

                if is_sharded:
                    for param in submodule.parameters(recurse=False):
                        param_id = id(param)
                        if param_id not in self._shards:
                            param.data = param.data.chunk(self.world_size, dim=0)[self.rank].contiguous().clone()
                            self._shards[param_id] = param.data
                    
                if is_sharded:
                    submodule.register_forward_pre_hook(self._forward_pre_hook)
                    submodule.register_forward_hook(self._forward_post_hook)

                    if isinstance(submodule, Linear):
                        submodule.register_full_backward_pre_hook(self._backward_pre_hook)

                for param in submodule.parameters(recurse=False):
                    param_id = id(param)
                    if param.requires_grad and param_id not in hook_params:
                        param.register_post_accumulate_grad_hook(self._backward_post_hook(submodule))
                        hook_params.add(param_id)


    def gather_full_params(self) -> dict[str, torch.Tensor]:
        full_params = {}
        for module_name, submodule in self.module.named_modules():
            if isinstance(submodule, (Embedding, Linear)):
                for param_name, param in submodule.named_parameters(recurse=False):
                    full = torch.empty(param.data.shape[0] * self.world_size, *param.data.shape[1:], dtype=param.data.dtype, device=param.data.device)
                    dist.all_gather_into_tensor(full, param.data.contiguous())
                    full_params[module_name + "." + param_name] = full
            else:
                for param_name, param in submodule.named_parameters(recurse=False):
                    full_params[module_name + "." + param_name] = param
        return full_params


    def all_gather_params(self, module: torch.nn.Module) -> None:
        """
        all-gather module parameters, support low-precision communication
        """
        for param in module.parameters(recurse=False):
            shard = param.data
            module._fp32_shard = shard
            dtype = self.compute_dtype if self.compute_dtype is not None else shard.dtype
            send = shard.to(dtype).contiguous() # make tensor contiguous in memory
            full = torch.empty(send.shape[0] * self.world_size, *send.shape[1:], dtype=send.dtype, device=send.device)
            dist.all_gather_into_tensor(full, send)
            param.data = full
            param.grad = None

    def async_all_gather_params(self, module: torch.nn.Module) -> None:
        """
        asynchronously all-gather module parameters, support low-precision communication
        """
        pending = []
        for param in module.parameters(recurse=False):
            shard = self._shards[id(param)]
            dtype = self.compute_dtype or shard.dtype
            send = shard.to(dtype).contiguous() # make tensor contiguous in memory
            full = torch.empty(send.shape[0] * self.world_size, *send.shape[1:], dtype=send.dtype, device=send.device)
            work = dist.all_gather_into_tensor(full, send, async_op=True)
            pending.append((param, send, full, work))
        return pending

    def wait_all_gather(self, pending) -> None:
        for param, _, full, work in pending:
            work.wait()
            param.data = full

    def free_full_params(self, module: torch.nn.Module) -> None:
        for param in module.parameters(recurse=False):
            param.data = self._shards[id(param)]

    def _forward_pre_hook(self, module: torch.nn.Module, args: tuple) -> None:
        if self._forward_order is None:
            self._recorded_order.append(module)
            pending = self.async_all_gather_params(module)
            self.wait_all_gather(pending)
            return

        index = self._forward_index
        if index >= len(self._forward_order) or self._forward_order[index] is not module:
            raise RuntimeError("FSDP forward execution order changed")

        pending = self._pending_gather.pop(index)
        self.wait_all_gather(pending)

        self._forward_index += 1


    def _forward_post_hook(self, module: torch.nn.Module, args: tuple, output: torch.Tensor) -> None:
        self.free_full_params(module)

        if self._forward_order is None:
            return

        prefetch_index = self._forward_index + 1
        if prefetch_index < len(self._forward_order):
            next_module = self._forward_order[prefetch_index]
            pending = self.async_all_gather_params(next_module)
            self._pending_gather[prefetch_index] = pending

    def forward(self, *inputs, **kwargs):
        self._forward_index = 0
        self._recorded_order = []
        self._pending_gather = {}

        if self._forward_order is not None:
            for index in range(min(2, len(self._forward_order))):
                module = self._forward_order[index]
                pending = self.async_all_gather_params(module)
                self._pending_gather[index] = pending

        output = self.module(*inputs, **kwargs)

        if self._forward_order is None:
            self._forward_order = list(self._recorded_order)
        elif self._forward_index != len(self._forward_order):
            raise RuntimeError("FSDP forward execution order changed")

        return output

    def _finish_grad_work(self, item):
        param, _, work_output, work = item
        work.wait()
        param.grad = work_output.to(torch.float32)


    def _queue_grad_work(self, item):
        self._grad_works.append(item)

        if len(self._grad_works) > 2: # at most preserve 2 grads
            self._finish_grad_work(self._grad_works.pop(0))


    def _backward_pre_hook(self, module: torch.nn.Module, grad_output: tuple) -> None:
        self.all_gather_params(module)


    def _backward_post_hook(self, module: torch.nn.Module) -> None:
        if isinstance(module, (Embedding, Linear)):
            def hook(param):
                shard = self._shards[id(param)]
                param.data = shard
                if param.grad is None:
                    return 
                full_grad = param.grad.contiguous()
                dtype = self.compute_dtype if self.compute_dtype is not None else full_grad.dtype
                shard_grad = torch.empty_like(shard, dtype=dtype)
                work = dist.reduce_scatter_tensor(shard_grad, full_grad, op=dist.ReduceOp.AVG, async_op=True)
                self._queue_grad_work((param, full_grad, shard_grad, work))
        else:
            def hook(param):
                if param.grad is None:
                    return
                work = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
                self._queue_grad_work((param, param.grad, param.grad, work))
        return hook


    def finish_gradient_synchronization(self):
        while self._grad_works:
            self._finish_grad_work(self._grad_works.pop(0))


MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}
VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 512


def profile_loss(model, inputs, targets, mixed_precision):
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=mixed_precision,
    ):
        return cross_entropy(model(inputs), targets)


def profile_train_step(model, optimizer, inputs, targets, mixed_precision):
    optimizer.zero_grad(set_to_none=True)
    loss = profile_loss(model, inputs, targets, mixed_precision)
    loss.backward()
    model.finish_gradient_synchronization()
    optimizer.step()


def run_profile(rank, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group(args.backend, rank=rank, world_size=args.world_size)
    device = torch.device("cuda", rank)
    config = MODEL_CONFIGS[args.model_size]
    local_batch_size = args.batch_size // args.world_size

    torch.manual_seed(42 + rank)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        **config,
    ).to(device)
    full_num_parameters = sum(param.numel() for param in base_model.parameters())
    compute_dtype = torch.float16 if args.mixed_precision else None
    model = FSDP(base_model, compute_dtype=compute_dtype)
    memory = {"after_model_init": memory_snapshot(model)}

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    inputs = torch.randint(
        VOCAB_SIZE,
        (local_batch_size, CONTEXT_LENGTH),
        device=device,
    )
    targets = torch.randint(
        VOCAB_SIZE,
        (local_batch_size, CONTEXT_LENGTH),
        device=device,
    )

    optimizer.zero_grad(set_to_none=True)
    loss = profile_loss(model, inputs, targets, args.mixed_precision)
    loss.backward()
    model.finish_gradient_synchronization()
    memory["before_optimizer_step"] = memory_snapshot(model, optimizer)
    optimizer.step()
    memory["after_optimizer_step"] = memory_snapshot(model, optimizer)
    del loss

    for _ in range(args.warmup_steps):
        profile_train_step(
            model,
            optimizer,
            inputs,
            targets,
            args.mixed_precision,
        )
    dist.barrier()
    torch.cuda.synchronize()
    start = timeit.default_timer()
    for _ in range(args.steps):
        profile_train_step(
            model,
            optimizer,
            inputs,
            targets,
            args.mixed_precision,
        )
    torch.cuda.synchronize()
    iteration_ms = torch.tensor(
        (timeit.default_timer() - start) * 1000 / args.steps,
        device=device,
    )
    dist.all_reduce(iteration_ms, op=dist.ReduceOp.MAX)

    memory_per_rank = [None] * args.world_size if rank == 0 else None
    dist.gather_object({"rank": rank, "phases": memory}, memory_per_rank, dst=0)
    if rank == 0:
        result = {
            "config": {
                **vars(args),
                "model": {
                    "vocab_size": VOCAB_SIZE,
                    "context_length": CONTEXT_LENGTH,
                    **config,
                },
                "compute_dtype": "fp16" if compute_dtype is not None else "fp32",
            },
            "full_num_parameters": full_num_parameters,
            "local_num_parameters": sum(
                param.numel() for param in model.parameters()
            ),
            "memory": summarize_memory(memory, memory_per_rank),
            "iteration_ms": round(iteration_ms.item(), 3),
        }
        output = json.dumps(result, indent=2)
        print(output)
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).resolve().parents[1] / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        precision = "fp16" if args.mixed_precision else "fp32"
        output_path = output_dir / (
            f"fsdp_{args.model_size}_{args.backend}_{precision}_ws{args.world_size}.json"
        )
        output_path.write_text(output + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Profile FSDP memory and full-step iteration time."
    )
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4, help="Global batch size across all ranks.")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("FSDP profiling requires CUDA for timing and memory measurements")

    mp.spawn(run_profile, args=(args,), nprocs=args.world_size, join=True)

"""
uv run nsys profile \
  -o results/fsdp_small_nccl_fp16 \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,nccl \
  --pytorch=functions-trace \
  --sample=none \
  --wait=all \
  -- python -m cs336_systems.fsdp \
    --model-size small \
    --backend nccl \
    --world-size 2 \
    --batch-size 4 \
    --mixed-precision \
    --output-dir results
"""

if __name__ == "__main__":
    main()
