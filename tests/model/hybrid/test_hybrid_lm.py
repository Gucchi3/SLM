# @file test_hybrid_lm.py
# @brief Tests of the loop of model/hybrid/hybrid_lm.py against the blocks run one by one by hand
# @author Kohei Yamaguchi, Claude

import pytest
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
            x, _ = block(x, rope, None)
        reference = model.head(model.norm(x))
        
        # The same weights with no loop, and with loops of times 1
        for loop in (LoopSpec(), LoopSpec(start=1, end=4, times=1), LoopSpec(start=0, end=5, times=1, reinject=True, kv_share=True)):
            model.loop = loop
            assert torch.equal(model(ids, ids), reference), loop
    return None


@pytest.mark.parametrize("reinject", [False, True])
def test_hybrid_lm_loop_matches_unrolled_blocks(reinject: bool) -> None:
    '''
    #* A loop over blocks 1 to 3 run 3 times is the same as those blocks written out 3 times, with the loop input added again after the first pass when reinject is set.
    '''
    
    # The blocks run by hand: block 0, then blocks 1, 2, 3 three times, then block 4
    torch.manual_seed(1)
    ids   = torch.randint(0, SPEC.vocab_size, (2, 19))
    model = HybridLm(SPEC, LoopSpec(start=1, end=4, times=3, reinject=reinject)).eval()
    rope  = (model.rope_cos[:19], model.rope_sin[:19])
    with torch.no_grad():
        x, _  = model.blocks[0](model.embed(ids), rope, None)
        start = x
        for step in range(3):
            if reinject and step > 0:
                x = x + start
            for index in (1, 2, 3):
                x, _ = model.blocks[index](x, rope, None)
        x, _      = model.blocks[4](x, rope, None)
        reference = model.head(model.norm(x))
        
        # The loop gives the same logits, and differs from one pass (so the loop really ran)
        assert torch.equal(model(ids, ids), reference)
        assert not torch.allclose(model(ids, ids, loops=1), reference, atol=1e-4)
    return None


def test_hybrid_lm_kv_share_uses_kv_of_first_pass() -> None:
    '''
    #* With kv_share, the attention blocks of the loop (a and w) use the k and v of their first pass in the later passes.
    '''
    
    # The blocks run by hand, keeping the k and v of the first pass of blocks 2 (a) and 3 (w)
    torch.manual_seed(2)
    ids   = torch.randint(0, SPEC.vocab_size, (1, 17))
    model = HybridLm(SPEC, LoopSpec(start=1, end=4, times=3, kv_share=True)).eval()
    rope  = (model.rope_cos[:17], model.rope_sin[:17])
    kept  = {}
    with torch.no_grad():
        x, _ = model.blocks[0](model.embed(ids), rope, None)
        for _ in range(3):
            for index in (1, 2, 3):
                x, kv = model.blocks[index](x, rope, kept.get(index))
                if index in (2, 3) and index not in kept:
                    kept[index] = kv
        x, _      = model.blocks[4](x, rope, None)
        reference = model.head(model.norm(x))
        
        # The same logits, and not the same as the loop without kv_share
        assert torch.equal(model(ids, ids), reference)
        model.loop = LoopSpec(start=1, end=4, times=3)
        assert not torch.allclose(model(ids, ids), reference, atol=1e-4)
    return None


def test_hybrid_lm_backprop_keeps_gradients_of_last_passes() -> None:
    '''
    #* With backprop 2 of 4 passes, the gradients are the same as those of the loop with the output of pass 2 detached.
    '''
    
    # The gradients of the model, with reinject so the loop input still learns through the last passes
    torch.manual_seed(3)
    ids   = torch.randint(0, SPEC.vocab_size, (2, 15))
    model = HybridLm(SPEC, LoopSpec(start=1, end=4, times=4, reinject=True, backprop=2)).train()
    model(ids, ids).square().mean().backward()
    grads = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
    model.zero_grad(set_to_none=True)
    
    # The reference: the same loop by hand, with the output of the first 2 passes detached
    rope  = (model.rope_cos[:15], model.rope_sin[:15])
    x, _  = model.blocks[0](model.embed(ids), rope, None)
    start = x
    for step in range(4):
        if step == 2:
            x = x.detach()
        if step > 0:
            x = x + start
        for index in (1, 2, 3):
            x, _ = model.blocks[index](x, rope, None)
    x, _ = model.blocks[4](x, rope, None)
    model.head(model.norm(x)).square().mean().backward()
    reference = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
    model.zero_grad(set_to_none=True)
    
    # The same gradients for every weight, and not the same as with all passes learning
    assert grads.keys() == reference.keys()
    assert all(torch.allclose(grads[name], reference[name], rtol=1e-5, atol=1e-7) for name in grads)
    model.loop = LoopSpec(start=1, end=4, times=4, reinject=True)
    model(ids, ids).square().mean().backward()
    full = dict(model.named_parameters())["blocks.1.qkv.weight"].grad
    assert full is not None and not torch.allclose(full, grads["blocks.1.qkv.weight"], atol=1e-6)
    return None


def test_hybrid_lm_number_of_passes() -> None:
    '''
    #* In training the loop runs a random number of times from train_min to train_max, in evaluation it runs times times, and the loops argument comes first in both.
    '''
    
    # Count the passes with a hook on the first block of the loop
    model  = HybridLm(SPEC, LoopSpec(start=1, end=3, times=5, train_min=2, train_max=4))
    counts = []
    model.blocks[1].register_forward_hook(lambda module, inputs, output: counts.append(1))
    ids    = torch.randint(0, SPEC.vocab_size, (1, 6))
    
    # Run once and return the number of passes
    def passes(loops: int | None) -> int:
        '''
        #* Return the number of passes of the loop in one forward pass.
        '''
        
        counts.clear()
        with torch.no_grad():
            model(ids, ids, loops)
        return len(counts)
    
    # Training: every number from 2 to 4, and nothing else. Evaluation: always times. loops: the given number
    model.train()
    assert {passes(None) for _ in range(60)} == {2, 3, 4}
    assert passes(7) == 7
    model.eval()
    assert {passes(None) for _ in range(5)} == {5}
    assert passes(1) == 1
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
