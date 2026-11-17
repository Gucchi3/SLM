# @file test_gated_delta.py
# @brief Tests of model/hybrid/gated_delta.py against the gated delta rule run one token at a time in float64
# @author Kohei Yamaguchi, Claude

from torch.nn import functional as F

import pytest
import torch

from model.hybrid.gated_delta import CHUNK, gated_delta_chunk



@pytest.mark.parametrize("tokens", [1, 5, CHUNK, CHUNK + 6, 2 * CHUNK + 1])
def test_gated_delta_chunk_matches_token_by_token(tokens: int) -> None:
    '''
    #* The chunked outputs are the same as S_t = exp(g_t) S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T and o_t = S_t q_t, run one token at a time in float64.
    '''
    
    # Inputs as the "g" block makes them: q and k L2-normalized (q scaled), g at most 0, beta in (0, 1), with value and key sizes that differ
    torch.manual_seed(tokens)
    batch, heads, dk, dv = 2, 3, 8, 6
    q    = F.normalize(torch.randn(batch, heads, tokens, dk), dim=-1) * dk**-0.5
    k    = F.normalize(torch.randn(batch, heads, tokens, dk), dim=-1)
    v    = torch.randn(batch, heads, tokens, dv)
    g    = -torch.rand(batch, heads, tokens) * 0.5
    beta = torch.rand(batch, heads, tokens)
    assert (g < 0).any() and (beta > 0.1).any()
    
    # The rule one token at a time (S is dv x dk)
    state     = torch.zeros(batch, heads, dv, dk, dtype=torch.float64)
    eye       = torch.eye(dk, dtype=torch.float64)
    reference = []
    for t in range(tokens):
        kt, vt, qt = k[:, :, t].double(), v[:, :, t].double(), q[:, :, t].double()
        bt         = beta[:, :, t].double()[..., None, None]
        state      = g[:, :, t].double().exp()[..., None, None] * state @ (eye - bt * kt[..., :, None] * kt[..., None, :]) + bt * vt[..., :, None] * kt[..., None, :]
        reference.append((state @ qt[..., :, None])[..., 0])
    reference = torch.stack(reference, dim=2)
    
    # The same outputs (float32 error only)
    out = gated_delta_chunk(q, k, v, g, beta)
    assert out.shape == (batch, heads, tokens, dv)
    assert torch.allclose(out.double(), reference, atol=1e-5), (out.double() - reference).abs().max()
    return None
