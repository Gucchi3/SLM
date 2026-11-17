# @file optimizer.py
# @brief 最適化器を作る

##########################################################
'''
最適化器の追加
1. Optimizer を返す build_xxx_optimizer(config, model) を書く。
2. OPTIMIZER_REGISTRY に "名前: build_xxx_optimizer" を足す。
3. config.train.optimizer.name をその名前にする。
'''
##########################################################

from collections.abc import Callable
from torch           import nn
from torch.optim     import AdamW, Optimizer
from types           import SimpleNamespace
from typing          import Any

import torch

from utils.exceptions import Error
from .muon_adamw      import MuonAdamW

OptimizerBuilder = Callable[[SimpleNamespace, nn.Module], Optimizer]



def build_adamw_optimizer(config: SimpleNamespace, model: nn.Module) -> Optimizer:
    '''
    #* AdamW を作る。weight decay は重み行列だけに掛け (バイアス、正規化の重み、埋め込みには掛けない)、埋め込みの学習率は embedding_lr_scale 倍にする。
    #! CUDA グラフを使うときは、各グループの学習率をデバイス上のテンソルにする。グラフを再生する合間に、スケジューラーが変えられるようにするため。
    #! betas は AdamW の 2 つの移動平均の減衰率 (例: [0.9, 0.999]、PyTorch の既定値)。embedding_lr_scale が 1.0 なら、埋め込みもほかと同じ学習率になる。
    '''
    
    # パラメータを、重み行列 (weight decay あり)、埋め込み、そのほか (バイアスと正規化の重み) に分ける
    embedding_ids = {id(parameter) for module in model.modules() if isinstance(module, nn.Embedding) for parameter in module.parameters()}
    parameters    = [parameter for parameter in model.parameters() if parameter.requires_grad]
    decay         = [parameter for parameter in parameters if parameter.ndim >= 2 and id(parameter) not in embedding_ids]
    embeddings    = [parameter for parameter in parameters if id(parameter) in embedding_ids]
    no_decay      = [parameter for parameter in parameters if parameter.ndim < 2 and id(parameter) not in embedding_ids]
    
    # AdamW を作る (CUDA では fused、CUDA グラフのために capturable)
    settings      = config.train.optimizer
    device        = parameters[0].device
    capturable    = config.train.cuda_graph and device.type == "cuda"
    learning_rate = settings.learning_rate
    decay_lr      = torch.tensor(learning_rate, device=device) if capturable else learning_rate
    embedding_lr  = torch.tensor(learning_rate * settings.embedding_lr_scale, device=device) if capturable else learning_rate * settings.embedding_lr_scale
    no_decay_lr   = torch.tensor(learning_rate, device=device) if capturable else learning_rate
    groups        = [{"params": decay, "weight_decay": settings.weight_decay, "lr": decay_lr}, {"params": embeddings, "weight_decay": 0.0, "lr": embedding_lr}, {"params": no_decay, "weight_decay": 0.0, "lr": no_decay_lr}]
    return AdamW(groups, betas=(settings.betas[0], settings.betas[1]), fused=device.type == "cuda", capturable=capturable)


def build_muon_optimizer(config: SimpleNamespace, model: nn.Module) -> Optimizer:
    '''
    #* 安定版の Muon を作る: Newton-Schulz の直交化、muon_momentum の Nesterov モーメンタム、重み行列への切り離した weight decay。埋め込みとそのほかには AdamW を使う。
    #? Keller Jordan の Muon の SingleDeviceMuonWithAuxAdam と同じ更新。Tiny-ML Leaderboard の Muon のモデルの多くがこれを使う。グループは _optimizer_muon_groups() で作る。
    #! learning_rate、betas、embedding_lr_scale は AdamW のグループ用。muon_learning_rate (例: 0.02) と weight_decay (例: 0.01) は重み行列用。
    '''
    
    muon = {"orthogonalization": "newton_schulz", "momentum_start": config.train.optimizer.muon_momentum, "momentum_warmup_steps": 0, "normuon": False, "beta2": 0.0, "cautious": False}
    return MuonAdamW(_optimizer_muon_groups(config, model, muon))


def build_normuon_optimizer(config: SimpleNamespace, model: nn.Module) -> Optimizer:
    '''
    #* 最新の Muon を作る: Polar Express の直交化、NorMuon、cautious weight decay、muon_momentum_start から muon_momentum へ上がっていくモーメンタム。埋め込みとそのほかには AdamW を使う。
    #? modded-nanogpt (NanoGPT speedrun) の 2025 年終わりから 2026 年の Muon から、多くの GPU 用の部分を除いたもの。Polar Express: arXiv 2505.16932。NorMuon: arXiv 2510.05491。Cautious weight decay: arXiv 2510.12402。
    #! build_muon_optimizer() のキーは同じように使う。normuon_beta2 は各行の 2 乗の移動平均の減衰率。muon_momentum_warmup_steps が 0 なら、モーメンタムは muon_momentum のまま。
    '''
    
    settings = config.train.optimizer
    muon     = {"orthogonalization": "polar_express", "momentum_start": settings.muon_momentum_start, "momentum_warmup_steps": settings.muon_momentum_warmup_steps, "normuon": True, "beta2": settings.normuon_beta2, "cautious": True}
    return MuonAdamW(_optimizer_muon_groups(config, model, muon))


def _optimizer_muon_groups(config: SimpleNamespace, model: nn.Module, muon: dict[str, Any]) -> list[dict[str, Any]]:
    '''
    #* MuonAdamW のグループを返す: 重み行列には Muon、埋め込みには AdamW (学習率は learning_rate の embedding_lr_scale 倍)、そのほか (バイアス、正規化の重み、LoRA の行列) には AdamW。
    #? build_muon_optimizer() と build_normuon_optimizer() が使う。muon には 2 つで違う設定が入る。
    #! 重み行列は、埋め込み以外の 2 次元以上のパラメータ。だから埋め込みと共有しない出力層も Muon になる。AdamW のグループには weight decay を掛けない。
    #! 名前に "lora" を含む行列は AdamW にする。Muon は細い行列 (288 x 4) の更新に sqrt(行 / 列) を掛ける。Glint-2 ではこれで LoRA の重みが大きくなり、x と LoRA の出力がお互いを大きくし合って、損失が NaN になった。
    #! CUDA グラフを使うときは、各学習率をデバイス上のテンソルにする。グラフを再生する合間に、スケジューラーが変えられるようにするため。
    '''
    
    # パラメータを、重み行列、埋め込み、そのほかに分ける
    embedding_ids = {id(parameter) for module in model.modules() if isinstance(module, nn.Embedding) for parameter in module.parameters()}
    named         = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameters    = [parameter for _, parameter in named]
    matrices      = [parameter for name, parameter in named if parameter.ndim >= 2 and id(parameter) not in embedding_ids and "lora" not in name]
    embeddings    = [parameter for parameter in parameters if id(parameter) in embedding_ids]
    others        = [parameter for name, parameter in named if id(parameter) not in embedding_ids and (parameter.ndim < 2 or "lora" in name)]
    
    # 各グループの学習率を作る (CUDA グラフのためにデバイス上のテンソルにする)
    settings   = config.train.optimizer
    device     = parameters[0].device
    capturable = config.train.cuda_graph and device.type == "cuda"
    rates      = [torch.tensor(rate, device=device) if capturable else rate for rate in (settings.muon_learning_rate, settings.learning_rate * settings.embedding_lr_scale, settings.learning_rate)]
    
    # グループを作る (build_adamw_optimizer() と同じく重み行列を先にし、履歴に Muon の学習率が残るようにする)
    adamw = {"kind": "adamw", "betas": (settings.betas[0], settings.betas[1]), "eps": 1e-8, "weight_decay": 0.0}
    return [{"params": matrices, "kind": "muon", "lr": rates[0], "weight_decay": settings.weight_decay, "momentum": settings.muon_momentum, **muon}, {"params": embeddings, "lr": rates[1], **adamw}, {"params": others, "lr": rates[2], **adamw}]


OPTIMIZER_REGISTRY: dict[str, OptimizerBuilder] = {
    "adamw"  : build_adamw_optimizer,
    "muon"   : build_muon_optimizer,
    "normuon": build_normuon_optimizer,
}


def build_optimizer(config: SimpleNamespace, model: nn.Module) -> Optimizer:
    '''
    #* config.train.optimizer.name の最適化器を作る。
    #? utils/trainer/pipeline.py から呼ぶ。
    '''
    
    name = config.train.optimizer.name
    if name not in OPTIMIZER_REGISTRY:
        raise Error("Unknown optimizer.", detail=f"train.optimizer.name: {name!r}\nAvailable: {', '.join(OPTIMIZER_REGISTRY)}", hint="Add the optimizer to OPTIMIZER_REGISTRY in utils/trainer/optimization/optimizer.py.", code=11)
    return OPTIMIZER_REGISTRY[name](config, model)
