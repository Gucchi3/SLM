# @file scheduler.py
# @brief スケジューラーを作る

##########################################################
'''
スケジューラーの追加
1. LRScheduler を返す build_xxx_scheduler(config, optimizer, total_steps) を書く。重みを更新するたびに 1 ステップ進む。
2. SCHEDULER_REGISTRY に "名前: build_xxx_scheduler" を足す。
3. config.train.scheduler.name をその名前にする。
'''
##########################################################

from collections.abc          import Callable
from torch.optim              import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from types                    import SimpleNamespace

import math

from utils.exceptions import Error

SchedulerBuilder = Callable[[SimpleNamespace, Optimizer, int], LRScheduler]



def build_cosine_annealing_scheduler(config: SimpleNamespace, optimizer: Optimizer, total_steps: int) -> LRScheduler:
    '''
    #* 学習率を warmup_steps の間は直線で上げ、その後はコサインに沿って最後のステップで最小値まで下げるスケジューラーを作る。
    #! config の学習率は、ウォームアップ後の一番上の値。1 ステップは重みの更新 1 回 (勾配を蓄積するときは、バッチ 1 つではない)。
    '''
    
    # 設定を読む
    warmup_steps  = config.train.scheduler.warmup_steps
    minimum_ratio = config.train.scheduler.minimum_learning_rate / config.train.optimizer.learning_rate
    decay_steps   = max(total_steps - warmup_steps, 1)
    
    # 各ステップの、一番上の値に対する学習率の比を返す
    return LambdaLR(optimizer, lambda step: (step + 1) / warmup_steps if step < warmup_steps else minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * min((step - warmup_steps) / decay_steps, 1.0))))


def build_wsd_scheduler(config: SimpleNamespace, optimizer: Optimizer, total_steps: int) -> LRScheduler:
    '''
    #* WSD (warmup、stable、decay) のスケジューラーを作る: 学習率を warmup_steps の間は直線で上げ、一番上のまま保ち、最後の decay_fraction の割合のステップで 1 - sqrt(進み具合) に沿って最小値まで下げる。
    #? Supra-Mini-v6 の train_model.py の get_wsd_lambda() と同じ形 (decay_fraction 0.2、最小値 0 を使う)。
    #! config の学習率は一番上の値。1 ステップは重みの更新 1 回 (勾配を蓄積するときは、バッチ 1 つではない)。
    '''
    
    # 設定を読み、下げ始めるステップを求める
    warmup_steps  = config.train.scheduler.warmup_steps
    minimum_ratio = config.train.scheduler.minimum_learning_rate / config.train.optimizer.learning_rate
    decay_start   = int(total_steps * (1 - config.train.scheduler.decay_fraction))
    decay_steps   = max(total_steps - decay_start, 1)
    
    # 各ステップの、一番上の値に対する学習率の比を返す
    return LambdaLR(optimizer, lambda step: (step + 1) / warmup_steps if step < warmup_steps else 1.0 if step < decay_start else minimum_ratio + (1 - minimum_ratio) * (1 - math.sqrt(min((step - decay_start) / decay_steps, 1.0))))


SCHEDULER_REGISTRY: dict[str, SchedulerBuilder] = {
    "cosine_annealing": build_cosine_annealing_scheduler,
    "wsd"             : build_wsd_scheduler,
}


def build_scheduler(config: SimpleNamespace, optimizer: Optimizer, total_steps: int) -> LRScheduler:
    '''
    #* config.train.scheduler.name のスケジューラーを作る。
    #? utils/trainer/pipeline.py から、学習全体の重みの更新回数を渡して呼ぶ。
    '''
    
    name = config.train.scheduler.name
    if name not in SCHEDULER_REGISTRY:
        raise Error("Unknown scheduler.", detail=f"train.scheduler.name: {name!r}\nAvailable: {', '.join(SCHEDULER_REGISTRY)}", hint="Add the scheduler to SCHEDULER_REGISTRY in utils/trainer/optimization/scheduler.py.", code=12)
    return SCHEDULER_REGISTRY[name](config, optimizer, total_steps)
