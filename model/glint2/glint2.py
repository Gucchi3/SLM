# @file glint2.py
# @brief 公式のチェックポイントにある coda のブロックを持つ Glint-2（パラメータ 1.71M のモデル）

from torch    import nn
from torch.nn import functional as F

import math
import torch

from .glint2_block import Glint2Block



VOCAB_SIZE       = 4096
DIM              = 96
NUM_HEADS        = 8
FFN_HIDDEN       = 2112
LORA_RANK        = 4
MAX_LOOPS        = 16
LOOPS            = 8
ATTENTION_WINDOW = 256
MAX_POSITIONS    = 1024
ROPE_BASE        = 10000.0
INIT_STD         = 0.02



class Glint2(nn.Module):
    '''
    #* ループのあと、最後の norm の前に、もう 1 つのブロック（coda）を持つ Glint-2。coda は自分の重みを持ち、ループの埋め込みも LoRA も使わずに 1 回だけ動く。
    #? model/build.py の build_model() が、model.name = "glint2" のときに作る。
    #! 公式の glint-2.pt はこのブロックを持つ（coda_layers = 1、パラメータ 1,710,049 個）。モデルカードにはループの外の層はないと書いてあり、generate.py はこのキーを古いものとして捨てる。この重みのとき、Glint-2.3-Mini の evaluate.py で測った WikiText-2 のバイト単位の perplexity は 3.126 で、公開された値は 3.127 である。coda がないと 3.501 になる。
    #! coda の場所と入力は、Glint-Research/Glint-2.3 の infer.py の forward() と同じにしている。ループのあと、ループの埋め込みも LoRA も使わず、final_norm の前に置く。
    '''
    
    cos   : torch.Tensor
    sin   : torch.Tensor
    window: torch.Tensor
    
    def __init__(self) -> None:
        '''
        #* Glint2 の層と coda のブロックを作り、coda の最初の重みを決める。
        #! coda の最初の重みは公開されていない。共有のブロックと同じにする。平均 0、std 0.02 の正規分布で、残差に足す out と down だけ std 0.02 / sqrt(2 * LOOPS) にする。
        '''
        
        super().__init__()
        self.embed      = nn.Embedding(VOCAB_SIZE, DIM)
        self.shared     = Glint2Block(DIM, NUM_HEADS, FFN_HIDDEN)
        self.lora_down  = nn.ModuleList(nn.Linear(DIM, LORA_RANK, bias=False) for _ in range(MAX_LOOPS))
        self.lora_up    = nn.ModuleList(nn.Linear(LORA_RANK, 3 * DIM, bias=False) for _ in range(MAX_LOOPS))
        self.loop_embed = nn.Embedding(MAX_LOOPS, DIM)
        self.final_norm = nn.RMSNorm(DIM)
        
        head_dim  = DIM // NUM_HEADS
        inv_freq  = 1.0 / ROPE_BASE ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        angles    = torch.outer(torch.arange(MAX_POSITIONS, dtype=torch.float32), inv_freq)
        positions = torch.arange(MAX_POSITIONS)
        distance  = positions[:, None] - positions[None, :]
        self.register_buffer("cos"   , angles.cos()                                  , persistent=False)
        self.register_buffer("sin"   , angles.sin()                                  , persistent=False)
        self.register_buffer("window", (distance >= 0) & (distance < ATTENTION_WINDOW), persistent=False)
        
        for name, parameter in self.named_parameters():
            if name.startswith("lora_up."):
                nn.init.zeros_(parameter)
            elif parameter.ndim == 2:
                residual = name in ("shared.out.weight", "shared.down.weight")
                nn.init.normal_(parameter, std=INIT_STD / math.sqrt(2 * LOOPS) if residual else INIT_STD)
        self.coda = Glint2Block(DIM, NUM_HEADS, FFN_HIDDEN)
        
        # coda の最初の重みを決める（norm の重みは 1 から始める）
        for name, parameter in self.coda.named_parameters():
            if parameter.ndim == 2:
                nn.init.normal_(parameter, std=INIT_STD / math.sqrt(2 * LOOPS) if name in ("out.weight", "down.weight") else INIT_STD)
        return None
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''
        #* 各位置で、次のトークンの logits を返す。形は (batch, tokens, VOCAB_SIZE)。
        #? utils/trainer/ の train.py と evaluate.py、utils/validator/ の generate.py と score.py から呼ぶ。
        #! Glint2 と同じく、attention_mask は使わない。CUDA graph のため、ここではテンソルの値を CPU 側で読まない。
        '''
        
        # 最初の位置の分の RoPE の表を取り出す。窓より遠いトークンがありうるときだけ、窓のマスクも取り出す
        tokens = input_ids.shape[1]
        cos    = self.cos[:tokens]
        sin    = self.sin[:tokens]
        window = self.window[:tokens, :tokens] if tokens > ATTENTION_WINDOW else None
        
        # 共有のブロックを LOOPS 回動かす
        x = self.embed(input_ids)
        for loop in range(LOOPS):
            x     = x + self.loop_embed.weight[loop]
            delta = self.lora_up[loop](self.lora_down[loop](x))
            x     = self.shared(x, cos, sin, delta, window)
        
        # coda を LoRA の変化なしで 1 回動かし、最後の隠れ状態を埋め込みの重みで logits に変える
        x = self.coda(x, cos, sin, x.new_zeros(()), window)
        return F.linear(self.final_norm(x), self.embed.weight)
