# @file hybrid_lm.py
# @brief The hybrid language model: embedding, the blocks of HybridSpec.mixers (with a loop over some of them), a final norm, and an output layer

from torch  import nn
from typing import cast

import torch

from .hybrid_block import HybridBlock
from .types        import HybridSpec, LoopSpec, MixerName



class HybridLm(nn.Module):
    '''
    #* Hold one hybrid model: token embedding, the blocks (the ones from loop.start to loop.end - 1 run again and again with the same weights), a final norm, and an output layer over the vocabulary.
    #? Made by build_model() in model/build.py from an entry of MODEL_REGISTRY and config.model.loop.
    #! forward() takes attention_mask only to match the other models of hitokoto. Padding is not supported, so every token sees all the tokens before it.
    #! The loop has no weights of its own, so a checkpoint loads with any loop settings. With times 1 (the default), the model is the same as the model with no loop.
    '''
    
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    
    def __init__(self, spec: HybridSpec, loop: LoopSpec) -> None:
        '''
        #* Make the embedding, the RoPE tables, the blocks, the final norm, and the output layer.
        '''
        
        # The embedding and the angles of RoPE
        super().__init__()
        self.spec  = spec
        self.loop  = loop
        self.embed = nn.Embedding(spec.vocab_size, spec.dim)
        half       = spec.head_dim // 2
        angles     = torch.arange(spec.max_length)[:, None] * spec.rope_theta ** (-torch.arange(half) / half)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)
        
        # The blocks, the final norm, and the output layer
        self.blocks = nn.ModuleList(HybridBlock(spec, cast(MixerName, mixer)) for mixer in spec.mixers)
        self.norm   = nn.LayerNorm(spec.dim) if spec.norm == "layer" else nn.RMSNorm(spec.dim)
        self.head   = nn.Linear(spec.dim, spec.vocab_size, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.head.weight , std=0.02)
        return None
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''
        #* Return the logits of the next token, of shape (batch, tokens, vocab_size).
        #? Called from train.py and evaluate.py in utils/trainer/, and from generate.py and score.py in utils/validator/.
        #! The inputs must have spec.max_length tokens or fewer.
        '''
        
        # The blocks in order, with the blocks of the loop (loop.start to loop.end - 1) run loop.times times
        loop   = self.loop
        tokens = input_ids.shape[1]
        rope   = (self.rope_cos[:tokens], self.rope_sin[:tokens])
        order  = [*range(loop.start), *list(range(loop.start, loop.end)) * loop.times, *range(loop.end, len(self.blocks))]
        x      = self.embed(input_ids)
        for index in order:
            x = self.blocks[index](x, rope)
        return self.head(self.norm(x))
