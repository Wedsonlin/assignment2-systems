"""
    implement flash attention v2 with pytorch autograd

    reference: http://arxiv.org/abs/2307.08691
"""

import math

import torch

class FlashAttentionV2Pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        is_causal: bool = False
    ) -> torch.Tensor:
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        b = Q.shape[0]
        N_q = Q.shape[1]
        N_k = K.shape[1]
        d_model = Q.shape[2]

        T_q = math.ceil(N_q / Q_TILE_SIZE) # don't consider out-of-bound
        Q = Q.reshape(b, T_q, Q_TILE_SIZE, d_model)
        T_k = math.ceil(N_k / K_TILE_SIZE)
        K = K.reshape(b, T_k, K_TILE_SIZE, d_model)
        V = V.reshape(b, T_k, K_TILE_SIZE, d_model)
        
        if is_causal:
            iota = torch.arange(N_q, device=Q.device)
            qi = iota.unsqueeze(-1)
            iota = torch.arange(N_k, device=K.device)
            kj = iota.unsqueeze(-1)
            causal_mask = qi >= kj  # (query, key)

        o = torch.empty((b, T_q, Q_TILE_SIZE, d_model), device=Q.device)
        l = torch.empty((b, T_q, Q_TILE_SIZE,), device=Q.device)
        for i in range(T_q):
            q_tile = Q[:, i]
            o_tile = torch.zeros((b, Q_TILE_SIZE, d_model), device=Q.device)
            l_tile = torch.zeros((b, Q_TILE_SIZE,), device=Q.device)
            m = torch.full((b, Q_TILE_SIZE,), float('-inf'), dtype=torch.float32, device=Q.device)

            for j in range(T_k):
                k_tile = K[:, j]
                v_tile = V[:, j]

                s_tile = (q_tile @ k_tile.transpose(1, 2)) / math.sqrt(d_model) # (b, Q_TILE_SIZE, K_TILE_SIZE)
                if is_causal:
                    mask_tile = causal_mask[i*Q_TILE_SIZE:(i+1)*Q_TILE_SIZE, j*K_TILE_SIZE:(j+1)*K_TILE_SIZE]
                    s_tile = torch.where(mask_tile, s_tile, float('-inf'))
                
                m_old = m
                m = torch.max(m, torch.amax(s_tile, dim=-1)) # (b, Q_TILE_SIZE)
                p = torch.exp(s_tile - m.unsqueeze(-1)) # (b, Q_TILE_SIZE, K_TILE_SIZE)
                exp_m_diff = torch.exp(m_old - m) # (b, Q_TILE_SIZE)
                l_tile = exp_m_diff * l_tile + p.sum(dim=2) # (b, Q_TILE_SIZE)
                o_tile = exp_m_diff.unsqueeze(-1) * o_tile + p @ v_tile
            
            o_tile =  torch.reciprocal(l_tile.unsqueeze(-1)) * o_tile
            l_tile = m + torch.log(l_tile)
            o[:, i] = o_tile
            l[:, i] = l_tile
        
        o = o.reshape(b, N_q, d_model)
        l = l.reshape(b, N_q)
        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, o, l)
        return o

    @staticmethod
    def backward(ctx, d_output):
        raise NotImplementedError