# @file hybrid_lm.py
# @brief The hybrid language model: embedding, the blocks of HybridSpec.mixers (with a loop over some of them), a final norm, and an output layer
# @author Kohei Yamaguchi, Claude

from torch  import nn
from typing import cast

import random
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
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, loops: int | None = None) -> torch.Tensor:
        '''
        #* Return the logits of the next token, of shape (batch, tokens, vocab_size).
        #? Called from train.py and evaluate.py in utils/trainer/, and from generate.py and score.py in utils/validator/.
        #! The inputs must have spec.max_length tokens or fewer. loops, when given, is the number of passes of the loop, in place of loop.times (and of the random number in training).
        #! The random number of passes in training comes from Python's random, so no tensor value is read on the CPU. A CUDA graph would keep the first number, so do not use it with train_min and train_max.
        '''
        
        # The number of passes: loops when given, random from train_min to train_max in training, or times
        loop   = self.loop
        tokens = input_ids.shape[1]
        rope   = (self.rope_cos[:tokens], self.rope_sin[:tokens])
        passes = loops if loops is not None else random.randint(loop.train_min, loop.train_max) if self.training and loop.train_max else loop.times
        
        # The blocks before the loop
        x = self.embed(input_ids)
        for block in self.blocks[:loop.start]:
            x, _ = block(x, rope, None)
        
        # The loop: each pass runs the same blocks. Only the last backprop passes keep gradients, the input comes back after the first pass with reinject, and kv_share keeps the k and v of the first pass
        start = x
        kept  = {}
        for step in range(passes):
            with torch.set_grad_enabled(torch.is_grad_enabled() and not (loop.backprop and step < passes - loop.backprop)):
                if loop.reinject and step > 0:
                    x = x + start
                for index in range(loop.start, loop.end):
                    x, kv = self.blocks[index](x, rope, kept.get(index))
                    if loop.kv_share and kv is not None:
                        kept.setdefault(index, kv)
        
        # The blocks after the loop, the final norm, and the output layer
        for block in self.blocks[loop.end:]:
            x, _ = block(x, rope, None)
        return self.head(self.norm(x))
