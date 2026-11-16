# @file hybrid_lm.py
# @brief hybrid 言語モデル: 埋め込み、HybridSpec.mixers のブロック (一部はループ)、最後の正規化、出力層

from torch    import nn
from torch.nn import functional as F
from typing   import Literal, cast

import torch

from .hybrid_block import HybridBlock
from .int_norm     import IntNorm
from .types        import HybridSpec, MixerName



class HybridLm(nn.Module):
    '''
    #* hybrid モデルを 1 つ持つ: トークン埋め込み (+ 学習する位置の表)、ブロック (loop.start から loop.end - 1 は同じ重みで何度も通す。周回ごとのループ埋め込みと LoRA も持てる)、最後の正規化、語彙への出力層。
    #? build_hybrid_model() (build_hybrid.py) が config.model.spec から作る。
    #! forward() の attention_mask は hitokoto のほかの部分と形を合わせるためだけに受け取る。パディングには対応しないので、各トークンは前の全トークンを見る。
    #! ループ埋め込みと LoRA を使わなければ、ループは自分の重みを持たないので、チェックポイントは times が違っても読み込める。使うときは times を学習時と同じにする。
    '''
    
    rope_global_cos: torch.Tensor
    rope_global_sin: torch.Tensor
    rope_window_cos: torch.Tensor
    rope_window_sin: torch.Tensor
    
    def __init__(self, spec: HybridSpec) -> None:
        '''
        #* 埋め込み、位置の表、RoPE の表、ブロック、ループの周回ごとの部品、最後の正規化、出力層を作る。
        '''
        
        # 埋め込み (rank があれば 2 段)、学習する位置の表 (embed.pos が learned のとき)
        super().__init__()
        self.spec     = spec
        self.embed    = nn.Embedding(spec.vocab_size, spec.embed.rank or spec.dim)
        self.embed_up = nn.Linear(spec.embed.rank, spec.dim, bias=False) if spec.embed.rank else None
        self.position = nn.Embedding(spec.max_length, spec.dim) if spec.embed.pos == "learned" else None
        
        # RoPE の角度の表を "a" と "w" で別々に作る。最初の part の割合の次元だけを theta で回し、残りは角度 0 (回さない)
        for name, attn in (("global", spec.attn_global), ("window", spec.attn_window)):
            pos    = attn.pos
            half   = attn.head_dim // 2
            turned = int(pos.part * half)
            angles = torch.arange(spec.max_length)[:, None] * torch.cat((pos.theta ** (-torch.arange(turned) / half), torch.zeros(half - turned)))
            self.register_buffer(f"rope_{name}_cos", angles.cos(), persistent=False)
            self.register_buffer(f"rope_{name}_sin", angles.sin(), persistent=False)
        
        # ブロック、最後の正規化、出力層 (rank があれば dim -> rank -> vocab の 2 段)
        kind           = spec.norm.type
        self.blocks    = nn.ModuleList(HybridBlock(spec, cast(MixerName, mixer), index) for index, mixer in enumerate(spec.mixers))
        self.norm      = nn.LayerNorm(spec.dim) if kind == "layer" else nn.RMSNorm(spec.dim) if kind == "rms" else IntNorm(cast(Literal["rmsp2", "dyt"], kind), spec.dim)
        self.head_down = nn.Linear(spec.dim, spec.embed.rank, bias=False) if spec.embed.rank and not spec.embed.tie else None
        self.head      = nn.Linear(spec.embed.rank or spec.dim, spec.vocab_size, bias=False)
        
        # ループの周回ごとの部品: ループ埋め込み (周回ごとに幅 dim、初期値 0) と、周回 x ループのブロックごとの低ランクの差分 (down は nn.Linear の初期値、up は 0)。学習の周回数の候補も持つ (学習のコードが読む)
        loop                  = spec.loop
        rank                  = loop.lora_rank
        turns                 = max((loop.times, *loop.times_train))
        looped                = [self.blocks[index] for index in range(loop.start, loop.end)] * turns if rank else []
        self.loop_times_train = loop.times_train
        self.loop_embed       = nn.Embedding(turns, spec.dim) if loop.embed else None
        self.lora_down        = nn.ModuleList(nn.Linear(spec.dim, rank, bias=False) for _ in looped)
        self.lora_up          = nn.ModuleList(nn.Linear(rank, cast(nn.Linear, block.qkv).out_features, bias=False) for block in looped)
        if self.loop_embed is not None:
            nn.init.zeros_(self.loop_embed.weight)
        for up in self.lora_up:
            nn.init.zeros_(cast(nn.Linear, up).weight)
        
        # 重みの初期値 (embed_up と head_down は nn.Linear の初期値のまま。アーキテクチャ探索と同じ)。tie なら出力層は埋め込みの重みを使う
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.head.weight , std=0.02)
        if self.position is not None:
            nn.init.normal_(self.position.weight, std=0.02)
        if spec.embed.tie:
            self.head.weight = self.embed.weight
        return None
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, times: int | None = None) -> torch.Tensor:
        '''
        #* 次のトークンのロジットを (batch, tokens, vocab_size) の形で返す。
        #? utils/trainer/ の train.py と evaluate.py、utils/validator/ の generate.py と score.py から呼ぶ。
        #! 入力は spec.max_length トークン以下にする。
        #! times はループの周回数で、None なら spec.loop.times。ループ埋め込みや LoRA を使うときは、times と times_train の一番大きい数以下にする。
        '''
        
        # 埋め込みと位置の表
        spec   = self.spec
        loop   = spec.loop
        tokens = input_ids.shape[1]
        x      = self.embed(input_ids)
        if self.embed_up is not None:
            x = self.embed_up(x)
        if self.position is not None:
            x = x + self.position.weight[:tokens]
        
        # 注意の種類ごとの RoPE の表 (type が none なら回さない)
        rope_global = (self.rope_global_cos[:tokens], self.rope_global_sin[:tokens]) if spec.attn_global.pos.type == "rope" else None
        rope_window = (self.rope_window_cos[:tokens], self.rope_window_sin[:tokens]) if spec.attn_window.pos.type == "rope" else None
        
        # ループの前のブロック
        memo  = cast(dict[int, tuple[torch.Tensor, torch.Tensor]], {})
        width = loop.end - loop.start
        for index in range(loop.start):
            x = self.blocks[index](x, rope_global if spec.mixers[index] == "a" else rope_window, memo, None)
        
        # ループのブロック (loop.start から loop.end - 1) を times 回 (None なら loop.times 回) 通す。周回の前にループ埋め込みを足し、各ブロックにはその周回の低ランクの差分を渡す
        for turn in range(loop.times if times is None else times):
            if self.loop_embed is not None:
                x = x + self.loop_embed.weight[turn]
            for offset in range(width):
                index = loop.start + offset
                slot  = turn * width + offset
                x     = self.blocks[index](x, rope_global if spec.mixers[index] == "a" else rope_window, memo, (self.lora_down[slot], self.lora_up[slot]) if self.lora_down else None)
        
        # ループの後のブロック
        for index in range(loop.end, len(self.blocks)):
            x = self.blocks[index](x, rope_global if spec.mixers[index] == "a" else rope_window, memo, None)
        
        # 最後の正規化と出力層 (rank があれば先に rank 次元へ縮める。tie では embed_up の転置を使う)
        h = self.norm(x)
        if self.embed_up is not None:
            h = self.head_down(h) if self.head_down is not None else F.linear(h, self.embed_up.weight.t())
        return self.head(h)
