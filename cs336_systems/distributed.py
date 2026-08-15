import os
import argparse
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend="gloo", # distributive communication backend
        rank=rank, # index of process
        world_size=world_size, # total number of proce*-+
    )
def distributed_demo(rank, world_size, data_size):
    setup(rank, world_size)
    data = torch.zeros(data_size * 1024 * 1024 // 4, dtype=torch.float32) # data_size MiB of data
    
    warmup_reps = 5
    for _ in range(warmup_reps):
        dist.all_reduce(data, async_op=False)

    dist.barrier()

    exe_reps = 10
    start = timeit.default_timer()
    for _ in range(exe_reps):
        dist.all_reduce(data, async_op=False)
    elapsed = timeit.default_timer() - start

    elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64)
    dist.reduce(elapsed_tensor, dst=0, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"Average time per all-reduce: {elapsed_tensor.item() / exe_reps * 1000:.3f} ms")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    data_size = args.data_size
    world_size = args.world_size

    mp.spawn(
        fn=distributed_demo,
        args=(world_size, data_size), # fn(rank, *args)
        nprocs=world_size,
        join=True, # wait for all processes to finish
    )