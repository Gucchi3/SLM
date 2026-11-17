# @file test_build.py
# @brief Tests of the hybrid models and the loop settings in model/build.py
# @author Kohei Yamaguchi, Claude

from types import SimpleNamespace

import pytest
import torch

from model            import build_model
from model.build      import MODEL_REGISTRY
from model.hybrid     import HybridLm, HybridSpec
from utils.exceptions import Error

# The hybrid models asked for, with their blocks and the values that differ from the standard model
HYBRID_MODELS = {
    "a4-kvh4"      : ("aaaa", {"kv_heads": 4}),
    "c3a"          : ("ccca", {}),
    "g3a"          : ("ggga", {}),
    "c3w"          : ("cccw", {}),
    "g3w"          : ("gggw", {}),
    "c3a-relu2-rms": ("ccca", {"mlp": "relu2", "norm": "rms"}),
    "c3w-relu2-rms": ("cccw", {"mlp": "relu2", "norm": "rms"}),
    "g3a-relu2-rms": ("ggga", {"mlp": "relu2", "norm": "rms"}),
    "g3w-relu2-rms": ("gggw", {"mlp": "relu2", "norm": "rms"}),
}



@pytest.mark.parametrize("name", list(HYBRID_MODELS))
def test_build_model_hybrid_names(name: str) -> None:
    '''
    #* Each hybrid name makes the model of its blocks and tags, on top of the standard model, and runs.
    '''
    
    # The registered shape: the standard values, with only the tags changed
    mixers, tags = HYBRID_MODELS[name]
    entry        = MODEL_REGISTRY[name]
    assert entry == HybridSpec(mixers=mixers, **tags)
    assert (HybridSpec(mixers="").vocab_size, HybridSpec(mixers="").max_length, HybridSpec(mixers="").dim, HybridSpec(mixers="").heads, HybridSpec(mixers="").head_dim) == (4096, 1024, 256, 4, 64)
    
    # The model has the blocks of the name, and gives logits over the vocabulary
    model = build_model(name, None, SimpleNamespace(start=0, end=3, times=2))
    assert isinstance(model, HybridLm)
    assert "".join(str(block.mixer) for block in model.blocks) == mixers
    with torch.no_grad():
        assert model.eval()(torch.zeros(1, 8, dtype=torch.long), torch.ones(1, 8)).shape == (1, 8, 4096)
    return None


@pytest.mark.parametrize(("name", "loop", "code"), [
    ("glint2", SimpleNamespace(times=2)                , 39),
    ("c3a"   , SimpleNamespace(times=2, train_max=3)   , 40),
    ("c3a"   , SimpleNamespace(start=2, end=5, times=2), 41),
    ("c3a"   , SimpleNamespace(start=3, end=1, times=2), 41),
    ("c3a"   , SimpleNamespace(start=0, end=3, times=0), 41),
])
def test_build_model_bad_loop(name: str, loop: SimpleNamespace, code: int) -> None:
    '''
    #* A loop on a model without loops, an unknown key (such as the removed train_max), and values out of range stop with their error codes.
    '''
    
    with pytest.raises(Error) as caught:
        build_model(name, None, loop)
    assert caught.value.code == code
    return None


def test_build_model_good_loop_edges() -> None:
    '''
    #* The loop over all blocks is allowed, and a missing key keeps its default value.
    '''
    
    model = build_model("c3w", None, SimpleNamespace(start=1, end=4))
    assert isinstance(model, HybridLm) and (model.loop.start, model.loop.end, model.loop.times) == (1, 4, 1)
    return None
