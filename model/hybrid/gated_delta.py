# @file gated_delta.py
# @brief gated delta rule (Gated DeltaNet) を系列全体についてチャンクごとに計算する (PyTorch だけで書く)

import torch

# 1 チャンクのトークン数
CHUNK = 64



def gated_delta_chunk(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    '''
    #* gated delta rule S_t = exp(g_t) S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T の出力 o_t = S_t q_t を、全トークンについて返す。
    #? HybridBlock ("g" のミキサー、hybrid_block.py) から呼ぶ。flash-linear-attention の chunk_gated_delta_rule_ref と同じ計算を PyTorch だけで書いたもの。
    #! q、k、v の形は (batch, heads, tokens, dim)。q と k は L2 正規化済みで、q はスケール済みにしておく。g (減衰の log、0 以下) と beta の形は (batch, heads, tokens)。
    #! 計算は float32 で行う。tokens は CHUNK の倍数でなくてもよい (足りない分は後ろを 0 で埋めて、最後に切り戻す)。
    '''
    
    # チャンク単位になるまで後ろを埋め、CHUNK トークンずつに切る (float32)
    dtype                    = q.dtype
    batch, heads, tokens, dk = q.shape
    dv                       = v.shape[-1]
    pad                      = (-tokens) % CHUNK
    q, k, v, g, beta         = (torch.nn.functional.pad(t.float(), (0, 0, 0, pad) if t.dim() == 4 else (0, pad)) for t in (q, k, v, g, beta))
    n                        = (tokens + pad) // CHUNK
    q, k, v                  = (t.view(batch, heads, n, CHUNK, t.shape[-1]) for t in (q, k, v))
    g, beta                  = (t.view(batch, heads, n, CHUNK) for t in (g, beta))
    
    # チャンクの中: 減衰の累積、トークン間の減衰、T = (I + 狭義下三角 (beta k k^T * 減衰))^-1
    g        = g.cumsum(-1)
    lower    = torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=q.device).tril()
    decay    = (g[..., :, None] - g[..., None, :]).masked_fill(~lower, -torch.inf).exp()
    kb       = k * beta[..., None]
    a        = (kb @ k.transpose(-1, -2) * decay).masked_fill(lower.logical_not() | torch.eye(CHUNK, dtype=torch.bool, device=q.device), 0)
    eye      = torch.eye(CHUNK, device=q.device).expand_as(a)
    t        = torch.linalg.solve_triangular(eye + a, eye, upper=False)
    value    = t @ (v * beta[..., None])
    k_decay  = t @ (kb * g.exp()[..., None])
    
    # チャンクをまたいで: 状態 S (dk x dv) を次のチャンクへ引き継ぐ
    state = q.new_zeros(batch, heads, dk, dv)
    out   = []
    for i in range(n):
        qi, ki, gi = q[:, :, i], k[:, :, i], g[:, :, i]
        new        = value[:, :, i] - k_decay[:, :, i] @ state
        inner      = (qi @ ki.transpose(-1, -2) * decay[:, :, i]).masked_fill(~lower, 0)
        out.append((qi * gi.exp()[..., None]) @ state + inner @ new)
        state      = state * gi[..., -1, None, None].exp() + (ki * (gi[..., -1, None] - gi).exp()[..., None]).transpose(-1, -2) @ new
    return torch.stack(out, dim=2).view(batch, heads, n * CHUNK, dv)[:, :, :tokens].to(dtype)
