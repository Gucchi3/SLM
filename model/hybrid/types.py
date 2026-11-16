# @file types.py
# @brief The shape of a hybrid model and the settings of its loop

from dataclasses import dataclass
from typing      import Literal

# The token mixer of a block: c = gated short convolution (LFM2), g = Gated DeltaNet, a = causal attention over all tokens, w = causal attention over the last window tokens
type MixerName = Literal["c", "g", "a", "w"]



@dataclass(frozen=True)
class HybridSpec:
    '''
    #* Hold the shape of one hybrid model. Each letter of mixers makes one block, and the blocks run in that order.
    #? Written in MODEL_REGISTRY of model/build.py, and given to HybridLm in hybrid_lm.py.
    #! The default values are the standard model: vocab 4096, length 1024, dim 256, attention of 4 heads x 64 with 1 kv head, GELU MLP of 1024, LayerNorm, RoPE, kernel 7 for "c" and 4 for "g", window 256.
    #! kv_heads must divide heads. The "g" blocks have gdn_heads heads with keys and values of gdn_dim each.
    '''
    
    mixers     : str
    dim        : int                                = 256
    heads      : int                                = 4
    kv_heads   : int                                = 1
    head_dim   : int                                = 64
    window     : int                                = 256
    mlp        : Literal["gelu", "relu2", "swiglu"] = "gelu"
    mlp_hidden : int                                = 1024
    norm       : Literal["layer", "rms"]            = "layer"
    c_kernel   : int                                = 7
    g_kernel   : int                                = 4
    gdn_heads  : int                                = 4
    gdn_dim    : int                                = 32
    rope_theta : float                              = 10000.0
    vocab_size : int                                = 4096
    max_length : int                                = 1024


@dataclass(frozen=True)
class LoopSpec:
    '''
    #* Hold the settings of the loop: the blocks from start to end - 1 run times times with the same weights, in training and in evaluation.
    #? Read from config.model.loop by build_model() in model/build.py, and given to HybridLm.
    #! times 1 (the default) runs each block once, the same as no loop.
    '''
    
    start: int = 0
    end  : int = 0
    times: int = 1
