# @file int_norm.py
# @brief 整数で計算しやすい正規化: 2 の累乗に丸めた RMSNorm と Dynamic Tanh

from torch  import nn
from typing import Literal

import torch



class IntNorm(nn.Module):
    '''
    #* 整数向けの正規化を 1 つ持つ。"rmsp2" は 1 / rms を 2 の累乗に丸めた RMSNorm (割り算がシフトになる)、"dyt" は Dynamic Tanh (weight x tanh(alpha x) + bias、統計量を使わない)。
    #? NormSpec.type が "rmsp2" か "dyt" のとき、HybridBlock と HybridLm が作る。"dyt" は Zhu ほか 2025 (Transformers without Normalization) に従い、alpha は 0.5 から始める。
    #! "rmsp2" は forward で log2(1 / rms) を一番近い整数に丸め、勾配は丸めが無いものとして通す (straight-through)。
    '''
    
    def __init__(self, kind: Literal["rmsp2", "dyt"], dim: int) -> None:
        '''
        #* weight を作る ("dyt" では bias と alpha も作る)。
        '''
        
        super().__init__()
        self.kind   = kind
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim)) if kind == "dyt" else None
        self.alpha  = nn.Parameter(torch.tensor(0.5)) if kind == "dyt" else None
        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        #* 正規化した x を同じ形で返す。
        '''
        
        # Dynamic Tanh
        if self.alpha is not None and self.bias is not None:
            return torch.tanh(self.alpha * x) * self.weight + self.bias
        
        # 1 / rms を 2 の累乗に丸めた RMSNorm
        inv   = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + torch.finfo(x.dtype).eps)
        shift = inv + (torch.exp2(torch.log2(inv).round()) - inv).detach()
        return x * shift * self.weight
