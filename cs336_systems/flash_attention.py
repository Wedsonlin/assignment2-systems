"""
implement flash attention v2 with pytorch autograd

reference: http://arxiv.org/abs/2307.08691
"""

import math

import torch
import triton
import triton.language as tl


def pytorch_backward(ctx, d_o):
    """PyTorch reference backward used by both FlashAttention frontends.

    The forward kernel stores the log-sum-exp in fp32. Reconstructing the
    probabilities therefore promotes the expression to fp32; cast it back to
    the input dtype before matrix multiplications so BF16 inputs do not produce
    mixed-dtype matmul errors.
    """
    Q, K, V, o, logsumexp = ctx.saved_tensors
    is_causal = ctx.is_causal
    if is_causal:
        iota = torch.arange(Q.shape[1], device=Q.device)
        qi = iota[..., None]
        iota = torch.arange(K.shape[1], device=K.device)
        kj = iota[None, ...]
        causal_mask = qi >= kj  # (query, key)

    d_model = Q.shape[-1]
    scale = 1.0 / math.sqrt(d_model)
    D = (o.float() * d_o.float()).sum(dim=-1)  # (b, N_q)
    S = Q @ K.transpose(1, 2) * scale
    if is_causal:
        S = torch.where(causal_mask, S, float("-inf"))
    P = torch.exp(S.float() - logsumexp.unsqueeze(-1)).to(Q.dtype)
    d_V = P.transpose(1, 2) @ d_o
    d_P = d_o @ V.transpose(1, 2)
    d_S = (P.float() * (d_P.float() - D.unsqueeze(-1))).to(Q.dtype)
    d_Q = d_S @ K * scale
    d_K = d_S.transpose(1, 2) @ Q * scale
    return d_Q, d_K, d_V, None


TRITON_FORWARD_CONFIGS = [
    triton.Config({"Q_TILE_SIZE": 16, "K_TILE_SIZE": 16}, num_warps=4, num_stages=2),
    triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 16}, num_warps=4, num_stages=2),
    triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 32}, num_warps=4, num_stages=2),
    triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 32}, num_warps=4, num_stages=3),
    triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_warps=8, num_stages=3),
    triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 32}, num_warps=8, num_stages=3),
]


class FlashAttentionV2Pytorch(torch.autograd.Function):
    """Small PyTorch reference kept for correctness tests and adapter compatibility."""

    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(Q.shape[-1])
        scores = Q @ K.transpose(-2, -1) * scale
        if is_causal:
            queries = torch.arange(Q.shape[-2], device=Q.device)[:, None]
            keys = torch.arange(K.shape[-2], device=K.device)[None, :]
            scores = torch.where(queries >= keys, scores, float("-inf"))

        logsumexp = torch.logsumexp(scores.float(), dim=-1)
        output = torch.softmax(scores, dim=-1) @ V
        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, output, logsumexp)
        return output

    @staticmethod
    def backward(ctx, d_o):
        return pytorch_backward(ctx, d_o)


@triton.autotune(
    configs=TRITON_FORWARD_CONFIGS,
    key=["N_QUERIES", "D", "is_causal", "ALLOW_TF32"],
    cache_results=True,
)
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    # Program indices, it's common to put main parallel dimension first
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    denominator = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    if is_causal:
        iota = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    num_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    if is_causal:
        # Entire K/V tiles to the right of this query tile are masked. Avoid
        # loading or multiplying them; the final partially visible tile still
        # uses the element-wise causal mask below.
        visible_keys = tl.minimum((query_tile_index + 1) * Q_TILE_SIZE, N_KEYS)
        num_key_tiles = tl.cdiv(visible_keys, K_TILE_SIZE)

    for i in range(num_key_tiles):
        k_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        if ALLOW_TF32:
            s = tl.dot(q_tile, tl.trans(k_tile), input_precision="tf32") * scale
        else:
            s = tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee") * scale
        if is_causal:
            iotb = i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask_tile = iota[:, None] >= iotb[None, :]
            s = tl.where(mask_tile, s, float("-inf"))

        old_m = m
        m = tl.maximum(m, tl.max(s, axis=-1))  # (Q_TILE_SIZE,)
        p = tl.exp(s - m[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)
        exp_m_diff = tl.exp(old_m - m)  # (Q_TILE_SIZE,)
        denominator = exp_m_diff * denominator + tl.sum(p, axis=-1)  # (Q_TILE_SIZE,)

        o = exp_m_diff[:, None] * o
        p = p.to(v_tile.dtype)
        if ALLOW_TF32:
            o = tl.dot(p, v_tile, acc=o, input_precision="tf32")
        else:
            o = tl.dot(p, v_tile, acc=o, input_precision="ieee")

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o = o / denominator[:, None]
    logsumexp = m + tl.log(denominator)

    o = o.to(O_ptr.type.element_ty)
    tl.store(O_block_ptr, o, boundary_check=(0, 1))
    tl.store(L_block_ptr, logsumexp, boundary_check=(0,))


class FlashAttentionV2Triton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        b = Q.shape[0]
        N_q = Q.shape[1]
        N_k = K.shape[1]
        d_model = Q.shape[2]
        scale = 1.0 / math.sqrt(d_model)

        o = torch.empty((b, N_q, d_model), dtype=Q.dtype, device=Q.device)
        logsumexp = torch.empty((b, N_q), dtype=torch.float32, device=Q.device)

        def grid(meta):
            return triton.cdiv(N_q, meta["Q_TILE_SIZE"]), b

        flash_fwd_kernel[grid](
            Q, K, V,
            o, logsumexp,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            logsumexp.stride(0), logsumexp.stride(1),
            N_q, N_k,
            scale,
            d_model,
            is_causal=is_causal,
            ALLOW_TF32=torch.get_float32_matmul_precision() != "highest",
        )

        ctx.save_for_backward(Q, K, V, o, logsumexp)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, d_o):
        return pytorch_backward(ctx, d_o)
