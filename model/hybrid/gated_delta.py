# @file gated_delta.py
# @brief The gated delta rule (Gated DeltaNet) over a whole sequence, computed in chunks with PyTorch only

import torch

# The number of tokens in one chunk
CHUNK = 64



def gated_delta_chunk(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    '''
    #* Return the outputs o_t = S_t q_t of the gated delta rule S_t = exp(g_t) S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T, for all tokens.
    #? Called by HybridBlock (the "g" mixer) in hybrid_block.py. The same math as chunk_gated_delta_rule_ref of flash-linear-attention, written for PyTorch only.
    #! q, k, v are (batch, heads, tokens, dim), with q and k already L2-normalized and q scaled; g (the log of the decay, at most 0) and beta are (batch, heads, tokens).
    #! The work is done in float32. tokens must be a multiple of CHUNK or shorter (shorter inputs are padded at the end and cut back).
    '''
    
    # Pad to whole chunks and cut into chunks of CHUNK tokens (float32)
    dtype                    = q.dtype
    batch, heads, tokens, dk = q.shape
    dv                       = v.shape[-1]
    pad                      = (-tokens) % CHUNK
    q, k, v, g, beta         = (torch.nn.functional.pad(t.float(), (0, 0, 0, pad) if t.dim() == 4 else (0, pad)) for t in (q, k, v, g, beta))
    n                        = (tokens + pad) // CHUNK
    q, k, v                  = (t.view(batch, heads, n, CHUNK, t.shape[-1]) for t in (q, k, v))
    g, beta                  = (t.view(batch, heads, n, CHUNK) for t in (g, beta))
    
    # Inside each chunk: the running decay, the decay between tokens, and T = (I + strictly lower (beta k k^T * decay))^-1
    g        = g.cumsum(-1)
    lower    = torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=q.device).tril()
    decay    = (g[..., :, None] - g[..., None, :]).masked_fill(~lower, -torch.inf).exp()
    kb       = k * beta[..., None]
    a        = (kb @ k.transpose(-1, -2) * decay).masked_fill(lower.logical_not() | torch.eye(CHUNK, dtype=torch.bool, device=q.device), 0)
    eye      = torch.eye(CHUNK, device=q.device).expand_as(a)
    t        = torch.linalg.solve_triangular(eye + a, eye, upper=False)
    value    = t @ (v * beta[..., None])
    k_decay  = t @ (kb * g.exp()[..., None])
    
    # Across chunks: carry the state S (dk x dv) from one chunk to the next
    state = q.new_zeros(batch, heads, dk, dv)
    out   = []
    for i in range(n):
        qi, ki, gi = q[:, :, i], k[:, :, i], g[:, :, i]
        new        = value[:, :, i] - k_decay[:, :, i] @ state
        inner      = (qi @ ki.transpose(-1, -2) * decay[:, :, i]).masked_fill(~lower, 0)
        out.append((qi * gi.exp()[..., None]) @ state + inner @ new)
        state      = state * gi[..., -1, None, None].exp() + (ki * (gi[..., -1, None] - gi).exp()[..., None]).transpose(-1, -2) @ new
    return torch.stack(out, dim=2).view(batch, heads, n * CHUNK, dv)[:, :, :tokens].to(dtype)
