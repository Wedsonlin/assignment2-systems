import argparse

import timeit
import numpy as np
import pandas as pd
import torch
import torch.cuda.nvtx as nvtx

import cs336_basics
from cs336_basics.model import annotated_scaled_dot_product_attention
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

args = argparse.ArgumentParser()

args.add_argument('--vocab_size', default=10000, type=int)
args.add_argument('--context_length', default=256, type=int)
args.add_argument('--d_model', default=512, type=int)
args.add_argument('--num_layers', default=4, type=int)
args.add_argument('--num_heads', default=16, type=int)
args.add_argument('--d_ff', default=1344, type=int)
args.add_argument('--rope_theta', default=100000, type=int)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = args.parse_args()
model = BasicsTransformerLM(
    vocab_size=args.vocab_size,
    context_length=args.context_length,
    d_model=args.d_model,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    rope_theta=args.rope_theta,
).to(device)

optimizer = AdamW(model.parameters())

dataset = np.random.randint(0,args.vocab_size,size=(1000,))

timer = timeit.default_timer

w = 5 # warm-up steps
n = 10 # profile steps
batch_size = 128

def forward(model, x):
    model(x)

def backward(loss):
    loss.backward()

def single_step(dataset, model, optimizer):
    x,gt = get_batch(dataset, batch_size=batch_size, context_length=args.context_length, device="cuda")

    start_time = timer()
    # nvtx.range_push(f"forwrad-bs{batch_size}")
    y = model(x)
    torch.cuda.synchronize()
    # nvtx.range_pop()
    end_time = timer()
    forward_time = end_time - start_time

    # train_loss = cross_entropy(y, gt)
    # optimizer.zero_grad()

    # start_time = timer()
    # # nvtx.range_push(f"backward-bs{batch_size}")
    # train_loss.backward()
    # torch.cuda.synchronize()
    # # nvtx.range_pop()
    # end_time = timer()
    # backward_time = end_time - start_time

    # # nvtx.range_push(f"optimizer step-bs{batch_size}")
    # optimizer.step()
    torch.cuda.synchronize()
    # nvtx.range_pop()

    return forward_time, 0

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
for _ in range(n):
    ft,bt = single_step(dataset, model, optimizer)
    w_forward_time.append(ft)
    w_backward_time.append(bt)

f_mean,f_std = cal_ave_and_std(w_forward_time)
b_mean,b_std = cal_ave_and_std(w_backward_time)

print("profile steps:")
print(f"forwrad: avg({f_mean}), std({f_std})")
print(f"backward: avg({b_mean}), std({b_std})")





    




