# @file test_hybrid_lm.py
# @brief model/hybrid/hybrid_lm.py のループを、ブロックを 1 つずつ手で動かした結果と比べるテスト
# @author Kohei Yamaguchi, Claude

import torch

from model.hybrid import HybridLm, HybridSpec, LoopSpec

# CPU でテストが速く終わるように、4 種類の mixer をすべて持つ小さなモデルを使う
SPEC = HybridSpec(mixers="gcawc", dim=32, heads=4, kv_heads=2, head_dim=8, window=4, mlp_hidden=48, gdn_heads=2, gdn_dim=8, vocab_size=50, max_length=128)



def test_hybrid_lm_without_loop_matches_blocks_in_order() -> None:
    '''
    #* ループがないときと、どこかに times 1 のループがあるときは、ブロックを順に 1 回ずつ動かしたのと同じ logits になる。
    '''
    
    # ブロックを手で動かす: 埋め込み、各ブロックを 1 回、最後の norm、出力の層
    torch.manual_seed(0)
    ids   = torch.randint(0, SPEC.vocab_size, (2, 21))
    model = HybridLm(SPEC, LoopSpec()).eval()
    rope  = (model.rope_cos[:21], model.rope_sin[:21])
    with torch.no_grad():
        x = model.embed(ids)
        for block in model.blocks:
            x = block(x, rope)
        reference = model.head(model.norm(x))
        
        # 同じ重みで、ループがないときと、times 1 のループがあるとき
        for loop in (LoopSpec(), LoopSpec(start=1, end=4, times=1), LoopSpec(start=0, end=5, times=1)):
            model.loop = loop
            assert torch.equal(model(ids, ids), reference), loop
    return None


def test_hybrid_lm_loop_matches_unrolled_blocks() -> None:
    '''
    #* ブロック 1〜3 を 3 回くり返すループは、学習のときも評価のときも、そのブロックを 3 回書き並べたのと同じになる。
    '''
    
    # ブロックを手で動かす: ブロック 0、次にブロック 1、2、3 を 3 回、次にブロック 4
    torch.manual_seed(1)
    ids   = torch.randint(0, SPEC.vocab_size, (2, 19))
    model = HybridLm(SPEC, LoopSpec(start=1, end=4, times=3))
    rope  = (model.rope_cos[:19], model.rope_sin[:19])
    with torch.no_grad():
        x = model.embed(ids)
        for index in (0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4):
            x = model.blocks[index](x, rope)
        reference = model.head(model.norm(x))
        
        # どちらのモードでも logits が同じで、1 回だけ通したものとは違う（ループが本当に動いた）
        assert torch.equal(model.train()(ids, ids), reference)
        assert torch.equal(model.eval()(ids, ids), reference)
        model.loop = LoopSpec(start=1, end=4, times=1)
        assert not torch.allclose(model(ids, ids), reference, atol=1e-4)
    return None


def test_hybrid_lm_window_sees_last_window_tokens() -> None:
    '''
    #* "w" のブロックだけのモデルでは、あるトークンの出力は、window 個前のトークンを変えても変わらず、window - 1 個前のトークンを変えると変わる。
    '''
    
    # window 4 のブロック 1 つ: 位置 10 のトークンは、位置 7〜10 のトークンを見る
    torch.manual_seed(4)
    spec  = HybridSpec(mixers="w", dim=32, heads=4, kv_heads=1, head_dim=8, window=4, mlp_hidden=48, vocab_size=50, max_length=64)
    model = HybridLm(spec, LoopSpec()).eval()
    ids   = torch.randint(1, spec.vocab_size, (1, 12))
    with torch.no_grad():
        base = model(ids, ids)[0, 10]
        far  = ids.clone()
        near = ids.clone()
        far[0, 6]  = 0
        near[0, 7] = 0
        assert torch.equal(model(far, far)[0, 10], base)
        assert not torch.allclose(model(near, near)[0, 10], base, atol=1e-6)
    return None
