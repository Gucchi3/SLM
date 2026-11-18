# @file test_gated_delta.py
# @brief model/hybrid/gated_delta.py を、float64 で 1 トークンずつ計算した gated delta rule と比べるテスト
# @author Kohei Yamaguchi, Claude

from torch.nn import functional as F

import pytest
import torch

from model.hybrid.gated_delta import CHUNK, gated_delta_chunk



@pytest.mark.parametrize("tokens", [1, 5, CHUNK, CHUNK + 6, 2 * CHUNK + 1])
def test_gated_delta_chunk_matches_token_by_token(tokens: int) -> None:
    '''
    #* chunk でまとめて計算した出力が、S_t = exp(g_t) S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T と o_t = S_t q_t を float64 で 1 トークンずつ計算したものと同じになる。
    '''
    
    # "g" のブロックが作るのと同じ入力: q と k は L2 正規化し（q は scale をかける）、g は 0 以下、beta は (0, 1) の中。value と key の大きさは違う値にする
    torch.manual_seed(tokens)
    batch, heads, dk, dv = 2, 3, 8, 6
    q    = F.normalize(torch.randn(batch, heads, tokens, dk), dim=-1) * dk**-0.5
    k    = F.normalize(torch.randn(batch, heads, tokens, dk), dim=-1)
    v    = torch.randn(batch, heads, tokens, dv)
    g    = -torch.rand(batch, heads, tokens) * 0.5
    beta = torch.rand(batch, heads, tokens)
    assert (g < 0).any() and (beta > 0.1).any()
    
    # 1 トークンずつ計算する rule（S の形は dv x dk）
    state     = torch.zeros(batch, heads, dv, dk, dtype=torch.float64)
    eye       = torch.eye(dk, dtype=torch.float64)
    reference = []
    for t in range(tokens):
        kt, vt, qt = k[:, :, t].double(), v[:, :, t].double(), q[:, :, t].double()
        bt         = beta[:, :, t].double()[..., None, None]
        state      = g[:, :, t].double().exp()[..., None, None] * state @ (eye - bt * kt[..., :, None] * kt[..., None, :]) + bt * vt[..., :, None] * kt[..., None, :]
        reference.append((state @ qt[..., :, None])[..., 0])
    reference = torch.stack(reference, dim=2)
    
    # 出力が同じになる（float32 の誤差の分だけ違ってよい）
    out = gated_delta_chunk(q, k, v, g, beta)
    assert out.shape == (batch, heads, tokens, dv)
    assert torch.allclose(out.double(), reference, atol=1e-5), (out.double() - reference).abs().max()
    return None
