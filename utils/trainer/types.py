# @file types.py
# @brief 学習の型

from dataclasses              import dataclass
from torch                    import nn
from torch.optim              import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import torch

from utils.trainer.optimization import LossKind



@dataclass(frozen=True)
class Metrics:
    '''
    #* 2 回の検証の間のバッチ、またはローダー 1 周分の損失と正解率を持つ。
    #! accuracy は飛ばさない (-100 でない) 正解だけを数える。
    '''
    
    loss    : float
    accuracy: float
    samples : int
    
    @property
    def perplexity(self) -> float:
        '''
        #* perplexity、exp(loss) を返す。損失が大きすぎるとき (学習が壊れた後など) はエラーにせず inf になる。
        '''
        
        return torch.tensor(self.loss, dtype=torch.float64).exp().item()


@dataclass(frozen=True)
class TrainingComponents:
    '''
    #* 損失とその種類、最適化器、スケジューラーをまとめる。
    '''
    
    loss_fn  : nn.Module
    loss_kind: LossKind
    optimizer: Optimizer
    scheduler: LRScheduler


@dataclass
class TrainState:
    '''
    #* 学習がどこまで進んだかを持つ: 重みの更新回数、train データの位置、検証の最良 (一番低い) perplexity、履歴。
    #? pipeline.py の pipeline_train() が、最初からか train.resume から作る。train_steps() と loop_train() が進め、loop_train() が train_state.pt に保存する。
    #! batch は、そのエポックですでに使った micro-batch の数。学習が変えるので frozen にはしない。
    '''
    
    step     : int
    epoch    : int
    batch    : int
    best_ppl : float
    history  : list[dict[str, int | float]]
