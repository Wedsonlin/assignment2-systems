import os
import argparse
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy


class NaiveDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        
        # broadcast parameters from rank 0 to all other ranks in the same order
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)


    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    @nvtx.range("gradient_synchronization")
    def finish_gradient_synchronization(self):
        """"
            Naive DDP implementation.
            Average gradients across all ranks.
        """
        # for param in self.module.parameters():
        #     if param.grad is not None:
        #         dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        
        grads_list = [param.grad for param in self.module.parameters() if param.grad is not None]
        flattened_grads = torch._utils._flatten_dense_tensors(grads_list)
        dist.all_reduce(flattened_grads, op=dist.ReduceOp.AVG)
        unflattened_grads = torch._utils._unflatten_dense_tensors(flattened_grads, grads_list)
        for dst, src in zip(grads_list, unflattened_grads):
            dst.copy_(src)

class OptimizedDDP(NaiveDDP):
    def __init__(self, module: torch.nn.Module):
        super().__init__(module)
        self.handles = []

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._reduce_gradient)

    @nvtx.range("gradient_synchronization")
    def _reduce_gradient(self, param: torch.nn.Parameter):
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
        self.handles.append(handle)

    def finish_gradient_synchronization(self):
        """
            Optimized DDP implementation.
            Reduce gradients across all ranks.
        """
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

def setup_process_group(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            local_rank  = rank % device_count
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError("Unable to find CUDA devices.")
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device

def ddp_train(
    ddp_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_iters: int,
    local_batch_size: int,
):
    context_length = ddp_model.module.config["context_length"]
    vocab_size = ddp_model.module.config["vocab_size"]
    training_times = []
    communication_times = []
    for iter in range(num_iters):
        x = torch.randint(0, vocab_size, (local_batch_size, context_length), device=device)
        y = torch.randint(0, vocab_size, (local_batch_size, context_length), device=device)
        
        if device != "cpu":
            torch.cuda.synchronize()
        start_training_time = timeit.default_timer()
        with nvtx.range("forward"):
            loss = cross_entropy(ddp_model(x), y)
        optimizer.zero_grad()
        with nvtx.range("backward"):
            loss.backward()
        
        if device != "cpu":
            torch.cuda.synchronize()
        start_communication_time = timeit.default_timer()
        ddp_model.finish_gradient_synchronization()
        if device != "cpu":
            torch.cuda.synchronize()
        end_communication_time = timeit.default_timer()

        optimizer.step()
        if device != "cpu":
            torch.cuda.synchronize()
        end_training_time = timeit.default_timer()
        training_times.append(end_training_time - start_training_time)
        communication_times.append(end_communication_time - start_communication_time)

    return training_times, communication_times

def benchmark_ddp(
    rank: int, 
    ddp,
    world_size: int, 
    backend: str,
    batch_size: int,
    ):
    device = setup_process_group(rank=rank, world_size=world_size, backend=backend)
    model = BasicsTransformerLM(
        # vocab_size=10000,
        # context_length=512,
        # d_model=2560,
        # d_ff=10240,
        # num_layers=32,
        # num_heads=32,
        vocab_size=10000,
        context_length=128,
        d_model=768,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
    ).to(device)
    ddp_model = ddp(model)

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

    local_batch_size = batch_size // world_size
    warmup_iters = 5
    _, _ = ddp_train(
        ddp_model,
        optimizer,
        device,
        num_iters=warmup_iters,
        local_batch_size=local_batch_size,
    )

    exe_iters = 10
    training_times, communication_times = ddp_train(
        ddp_model,
        optimizer,
        device,
        num_iters=exe_iters,
        local_batch_size=local_batch_size,
    )

    training_time = sum(training_times) / len(training_times)
    communication_time = sum(communication_times) / len(communication_times)
    gather_training_time = [0.0] * world_size
    gather_communication_time = [0.0] * world_size
    dist.all_gather_object(gather_training_time, training_time)
    dist.all_gather_object(gather_communication_time, communication_time)

    if rank == 0:
        avg_training_time = sum(gather_training_time) / len(gather_training_time)
        avg_communication_time = sum(gather_communication_time) / len(gather_communication_time)
        print(f"Training step time: {avg_training_time * 1000:.3f} ms")
        print(f"Communication time: {avg_communication_time * 1000:.3f} ms")
        print(f"Communication portion: {avg_communication_time / avg_training_time * 100:.3f}%")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--optimized", action="store_true")
    args = parser.parse_args()
    ddp = OptimizedDDP if args.optimized else NaiveDDP
    mp.spawn(
        benchmark_ddp,
        args=(ddp, args.world_size, args.backend, args.batch_size),
        nprocs=args.world_size,
        join=True,
    )
