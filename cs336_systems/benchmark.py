import argparse

import timeit
import numpy as np
import pandas as pd
import torch
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext

import cs336_basics
from cs336_basics.model import annotated_scaled_dot_product_attention
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

args = argparse.ArgumentParser()

args.add_argument('--context_length', default=256, type=int)
args.add_argument('--d_model', default=512, type=int)
args.add_argument('--num_layers', default=4, type=int)
args.add_argument('--num_heads', default=16, type=int)
args.add_argument('--d_ff', default=1344, type=int)
args.add_argument('--use_autocast', default=True, type=bool)
args.add_argument('--forward_only', default=False, type=bool)
args.add_argument('--model', default='small', type=str)
args.add_argument('--profile_memory', default=False, type=bool)

args = args.parse_args()

model_config = {
    'small': {
        'd_model': 768,
        'd_ff': 3072,
        'num_layers': 12,
        'num_heads': 12,
    },
    'medium': {
        'd_model': 1024,
        'd_ff': 4096,
        'num_layers': 24,
        'num_heads': 16,
    },
    'large': {
        'd_model': 1280,
        'd_ff': 5120,
        'num_layers': 36,
        'num_heads': 20,
    },
    'xl': {
        'd_model': 1600,
        'd_ff': 6400,
        'num_layers': 48,
        'num_heads': 25,
    },
    '2.7B':{
        'd_model': 2560,
        'd_ff': 10240,
        'num_layers': 32,
        'num_heads': 32,
    }
}

config = model_config[args.model]
print(f"using {args.model} model")
print(f"model config: {config}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")

model = BasicsTransformerLM(
    vocab_size=10000,
    context_length=args.context_length,
    d_model=config['d_model'],
    num_layers=config['num_layers'],
    num_heads=config['num_heads'],
    d_ff=config['d_ff'],
    rope_theta=10000,
).to(device)

optimizer = AdamW(model.parameters())

dataset = np.random.randint(0,10000,size=(10000,))

timer = timeit.default_timer

w = 5 # warm-up steps
n = 10 # profile steps
batch_size = 4
autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if args.use_autocast else nullcontext()

def single_step(dataset, model, optimizer):
    x,gt = get_batch(dataset, batch_size=batch_size, context_length=args.context_length, device="cuda")

    start_time = timer()
    nvtx.range_push(f"forwrad-bs{batch_size}")
    with autocast_ctx:
        y = model(x)
    torch.cuda.synchronize()
    nvtx.range_pop()
    end_time = timer()
    forward_time = end_time - start_time

    if args.forward_only:
        return forward_time, 0
    else:
        train_loss = cross_entropy(y, gt)
        optimizer.zero_grad()

        start_time = timer()
        nvtx.range_push(f"backward-bs{batch_size}")
        train_loss.backward()
        torch.cuda.synchronize()
        nvtx.range_pop()
        end_time = timer()
        backward_time = end_time - start_time

        nvtx.range_push(f"optimizer step-bs{batch_size}")
        optimizer.step()
        torch.cuda.synchronize()
        nvtx.range_pop()

        return forward_time, backward_time

def cal_ave_and_std(time_list):
    mean = np.mean(time_list)
    std = np.std(time_list)
    return mean, std

w_forward_time = []
w_backward_time = []
for _ in range(w):
    ft,bt = single_step(dataset, model, optimizer)
    w_forward_time.append(ft)
    w_backward_time.append(bt)

f_mean,f_std = cal_ave_and_std(w_forward_time)
b_mean,b_std = cal_ave_and_std(w_backward_time)

print("warm up steps:")
print(f"forwrad: avg({f_mean}), std({f_std})")
print(f"backward: avg({b_mean}), std({b_std})")


w_forward_time = []
w_backward_time = []
if args.profile_memory:
    print("profiling memory")
    torch.cuda.memory._record_memory_history(max_entries=1000000)
for _ in range(n):
    ft,bt = single_step(dataset, model, optimizer)
    w_forward_time.append(ft)
    w_backward_time.append(bt)
if args.profile_memory:
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enable=None)
f_mean,f_std = cal_ave_and_std(w_forward_time)
b_mean,b_std = cal_ave_and_std(w_backward_time)

print("profile steps:")
print(f"forwrad: avg({f_mean}), std({f_std})")
print(f"backward: avg({b_mean}), std({b_std})")





    




