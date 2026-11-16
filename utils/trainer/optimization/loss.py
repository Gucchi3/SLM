# @file loss.py
# @brief 損失関数を作り、正解との損失と蒸留の損失を計算する

##########################################################
'''
損失関数の追加
1. nn.Module を継承し、reduction="none" で正解ごとに 1 つの損失を返す損失のクラスを作る。
2. LOSS_REGISTRY に "名前: (クラス, 種類)" を足す。種類は "softmax" (正解ごとにクラスが 1 つ) か "sigmoid" (正解ごとにはい/いいえ)。
3. config.train.loss.name をその名前にする。
'''
##########################################################

from torch.nn import functional as F
from types    import SimpleNamespace
from typing   import Literal

import torch
import torch.nn as nn

from utils.exceptions      import Error
from .cross_entropy_z_loss import CrossEntropyZLoss

LossKind = Literal["softmax", "sigmoid"]

IGNORE_INDEX = -100

LOSS_REGISTRY: dict[str, tuple[type[nn.Module], LossKind]] = {
    "cross_entropy"       : (nn.CrossEntropyLoss , "softmax"),
    "cross_entropy_z_loss": (CrossEntropyZLoss   , "softmax"),
    "bce_with_logits"     : (nn.BCEWithLogitsLoss, "sigmoid"),
}



def build_loss_fn(config: SimpleNamespace) -> tuple[nn.Module, LossKind]:
    '''
    #* config.train.loss.name の損失関数を作り、その種類と一緒に返す。
    #? utils/trainer/pipeline.py から呼ぶ。
    #! config.train.loss のほかのキーは損失のクラスに渡す (label_smoothing など)。loss_label() が -100 の正解を飛ばすので、reduction は常に "none"。
    '''
    
    # 名前を確かめる
    options = vars(config.train.loss).copy()
    name    = options.pop("name")
    if name not in LOSS_REGISTRY:
        raise Error("Unknown loss.", detail=f"train.loss.name: {name!r}\nAvailable: {', '.join(LOSS_REGISTRY)}", hint="Add the loss to LOSS_REGISTRY in utils/trainer/optimization/loss.py.", code=10)
    
    # 正解ごとに 1 つの値を返す損失を作る
    loss_class, kind     = LOSS_REGISTRY[name]
    options["reduction"] = "none"
    return loss_class(**options), kind


def loss_label(logits: torch.Tensor, targets: torch.Tensor, loss_fn: nn.Module, kind: LossKind) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    #* -100 の正解を飛ばして、正解との平均の損失、当たった正解の数、正解の数を返す。
    #? utils/trainer/ の train.py (学習) と evaluate.py (評価) から呼ぶ。
    #! softmax: targets はクラスの番号で、logits は targets より 1 つ次元 (クラス) が多い。sigmoid: targets は 0 か 1 で、logits は targets と同じ形。
    #! CPU で値を読まないので、CUDA グラフの中でも動く。
    '''
    
    # 正解ごとに 1 行になるよう平らにする
    flat_logits  = logits.flatten(0, -2).float() if kind == "softmax" else logits.flatten().float()
    flat_targets = targets.flatten()
    valid        = flat_targets != IGNORE_INDEX
    count        = valid.sum()
    
    # 飛ばさない正解についての平均の損失 (飛ばす正解は BCE では 0 にしてから、マスクで除く)
    loss_targets = flat_targets if kind == "softmax" else flat_targets.float().clamp(min=0)
    loss         = (loss_fn(flat_logits, loss_targets) * valid).sum() / count.clamp(min=1)
    
    # 当たった正解を数える: softmax は一番大きいクラス、sigmoid は 0 のどちら側か
    matched = flat_logits.argmax(dim=-1) == flat_targets if kind == "softmax" else (flat_logits > 0) == (flat_targets > 0.5)
    correct = (matched & valid).sum()
    return loss, correct, count


def loss_distill(logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor, kind: LossKind, temperature: float) -> torch.Tensor:
    '''
    #* -100 の正解を飛ばして、温度でやわらげた生徒と教師の間の平均の蒸留損失を返す。
    #? utils/trainer/train.py から呼ぶ。
    #! softmax: クラスの確率の KL ダイバージェンス。sigmoid: 正解ごとの教師の「はい」の確率に対する BCE。
    #! 教師のロジットは生徒のロジットと同じ形にする。勾配の大きさを保つため、結果には temperature² を掛ける。
    '''
    
    # 正解ごとに 1 行になるよう平らにする
    flat_student = (logits.flatten(0, -2) if kind == "softmax" else logits.flatten()).float() / temperature
    flat_teacher = (teacher_logits.flatten(0, -2) if kind == "softmax" else teacher_logits.flatten()).float() / temperature
    valid        = targets.flatten() != IGNORE_INDEX
    
    # 正解ごとの損失を求め、飛ばさない正解について平均する
    if kind == "softmax":
        soft_by_target = F.kl_div(flat_student.log_softmax(dim=-1), flat_teacher.log_softmax(dim=-1), reduction="none", log_target=True).sum(dim=-1)
    else:
        soft_by_target = F.binary_cross_entropy_with_logits(flat_student, flat_teacher.sigmoid(), reduction="none")
    return (soft_by_target * valid).sum() / valid.sum().clamp(min=1)
