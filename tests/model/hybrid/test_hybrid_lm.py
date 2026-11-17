# @file test_hybrid_lm.py
# @brief Tests of the loop of model/hybrid/hybrid_lm.py against the blocks run one by one by hand
# @author Kohei Yamaguchi, Claude

import torch

from model.hybrid import HybridLm, HybridSpec, LoopSpec

# A small model with all four mixers, so the tests run fast on the CPU
SPEC = HybridSpec(mixers="gcawc", dim=32, heads=4, kv_heads=2, head_dim=8, window=4, mlp_hidden=48, gdn_heads=2, gdn_dim=8, vocab_size=50, max_length=128)



def test_hybrid_lm_without_loop_matches_blocks_in_order() -> None:
    '''
    #* With no loop, and with a loop of times 1 at any place, the logits are the same as the blocks run once in order.
    '''
    
    # The blocks run by hand: embedding, each block once, the final norm, and the output layer
    torch.manual_seed(0)
    ids   = torch.randint(0, SPEC.vocab_size, (2, 21))
    model = HybridLm(SPEC, LoopSpec()).eval()
    rope  = (model.rope_cos[:21], model.rope_sin[:21])
    with torch.no_grad():
        x = model.embed(ids)
        for block in model.blocks:
            x = block(x, rope)
        reference = model.head(model.norm(x))
        
        # The same weights with no loop, and with loops of times 1
        for loop in (LoopSpec(), LoopSpec(start=1, end=4, times=1), LoopSpec(start=0, end=5, times=1)):
            model.loop = loop
            assert torch.equal(model(ids, ids), reference), loop
    return None


def test_hybrid_lm_loop_matches_unrolled_blocks() -> None:
    '''
    #* A loop over blocks 1 to 3 run 3 times is the same as those blocks written out 3 times, in training and in evaluation.
    '''
    
    # The blocks run by hand: block 0, then blocks 1, 2, 3 three times, then block 4
    torch.manual_seed(1)
    ids   = torch.randint(0, SPEC.vocab_size, (2, 19))
    model = HybridLm(SPEC, LoopSpec(start=1, end=4, times=3))
    rope  = (model.rope_cos[:19], model.rope_sin[:19])
    with torch.no_grad():
        x = model.embed(ids)
        for index in (0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4):
            x = model.blocks[index](x, rope)
        reference = model.head(model.norm(x))
        
        # The same logits in both modes, and not the same as one pass (so the loop really ran)
        assert torch.equal(model.train()(ids, ids), reference)
        assert torch.equal(model.eval()(ids, ids), reference)
        model.loop = LoopSpec(start=1, end=4, times=1)
        assert not torch.allclose(model(ids, ids), reference, atol=1e-4)
    return None


def test_hybrid_lm_window_sees_last_window_tokens() -> None:
    '''
    #* In a model of "w" blocks only, the output at a token does not change with the token window places before it, and changes with the token window - 1 places before it.
    '''
    
    # One block with window 4: the token at 10 sees the tokens 7 to 10
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
