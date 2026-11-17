# @file cross_entropy_z_loss.py
# @brief z-loss 付きの交差エントロピー (ロジットが大きくなりすぎないようにする)

from torch    import nn
from torch.nn import functional as F
from typing   import Literal

import torch



class CrossEntropyZLoss(nn.Module):
    '''
    #* 交差エントロピー + z_weight * logsumexp(logits)² を持つ。後ろの項は PaLM の z-loss で、ロジットが大きな値へずれていくのを防ぐ。
    #? loss.py の LOSS_REGISTRY に "cross_entropy_z_loss" として登録する。Supra-Mini-v6 の train_model.py は z_weight 1e-4 を使う。
    #! 各正解の損失に z-loss が入るので、train と validation の損失 (とそこから求める perplexity) は、交差エントロピーだけのときより少し大きい。
    '''
    
    def __init__(self, z_weight: float, reduction: Literal["none"]) -> None:
        '''
        #* z-loss の重みと reduction を覚える。
        #! reduction は "none" (正解ごとに 1 つの損失) にする。loss.py の build_loss_fn() は必ずそう渡す。
        '''
        
        super().__init__()
        self.z_weight  = z_weight
        self.reduction = reduction
        return None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        '''
        #* 正解ごとの損失を返す: 交差エントロピー + z_weight x (そのロジットの logsumexp の 2 乗)。
        #! logits の形は (targets, classes)、targets の形は (targets,)。-100 の正解は z-loss だけになり、loss.py の loss_label() がマスクで除く。
        '''
        
        return F.cross_entropy(logits, targets, reduction=self.reduction) + self.z_weight * logits.logsumexp(dim=-1) ** 2
