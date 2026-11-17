# @file muon_adamw.py
# @brief MuonAdamW: 重み行列には Muon、ほかのパラメータには AdamW を使う 1 つの最適化器

from collections.abc import Callable
from typing          import Any, Literal

import torch

Orthogonalization = Literal["newton_schulz", "polar_express"]
Coefficients      = tuple[tuple[float, float, float], ...]

# X <- a X + b (X Xᵀ) X + c (X Xᵀ)² X の各ステップの (a, b, c)
NEWTON_SCHULZ_COEFFICIENTS: Coefficients = ((3.4445, -4.7750, 2.0315),) * 5
POLAR_EXPRESS_COEFFICIENTS: Coefficients = (
    (8.156554524902461 , -22.48329292557795 , 15.878769915207462 ),
    (4.042929935166739 , -2.808917465908714 , 0.5000178451051316 ),
    (3.8916678022926607, -2.772484153217685 , 0.5060648178503393 ),
    (3.285753657755655 , -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)

# 直交化の種類ごとの係数と、最初のスケールの安全のための余裕 (最大の特異値を 1 より小さく保つ)
ORTHOGONALIZATIONS: dict[Orthogonalization, tuple[Coefficients, float]] = {
    "newton_schulz": (NEWTON_SCHULZ_COEFFICIENTS, 0.0 ),
    "polar_express": (POLAR_EXPRESS_COEFFICIENTS, 0.02),
}



class MuonAdamW(torch.optim.Optimizer):
    '''
    #* 種類が "muon" のグループは Muon (モーメンタムの後、更新に一番近い直交行列にする) で、種類が "adamw" のグループは AdamW で更新する、1 つの最適化器を持つ。
    #? optimizer.py の build_muon_optimizer() (Keller Jordan の安定版の Muon) と build_normuon_optimizer() (modded-nanogpt の最新の Muon) が作る。
    #! Muon は隠れ層の重み行列だけに使う。埋め込み、出力層、LoRA の行列、1 次元のパラメータ (正規化の重み、バイアス) は "adamw" のグループに入れる。
    #! step() は CPU で値を読まず、ステップ数もデバイス上のテンソルで数えるので、CUDA グラフの中でも動く。lr は float か、スケジューラーが変えるデバイス上のテンソル。
    '''
    
    def __init__(self, param_groups: list[dict[str, Any]]) -> None:
        '''
        #* パラメータのグループを覚える。
        #! "muon" のグループには lr、weight_decay、momentum、momentum_start、momentum_warmup_steps、orthogonalization、normuon、beta2、cautious が要る。"adamw" のグループには lr、betas、eps、weight_decay が要る。
        '''
        
        super().__init__(param_groups, {})
        return None
    
    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> Any:
        '''
        #* すべてのパラメータを 1 回更新し、closure があればその損失を返す (なければ None)。
        #! Muon: Nesterov モーメンタム、直交化、NorMuon (normuon のとき)、sqrt(max(1, 行 / 列)) でのスケール、切り離した weight decay (cautious のときは、更新と重みの符号が同じところだけ)。
        #! AdamW: 勾配とその 2 乗の移動平均 (バイアス補正付き) と、切り離した weight decay。
        #! 戻り値の型は Any にする。Optimizer.step() の 2 つの overload (closure 無しなら None、有りなら float) のどちらにも合うようにするため。
        '''
        
        # ほかの PyTorch の最適化器と同じように、closure を実行して損失を求める
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # 勾配を持つパラメータをそれぞれ更新する
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                grad  = parameter.grad
                if group["kind"] == "muon":
                    # 最初のステップで状態を作る: モーメンタム、ステップ数、NorMuon では各行 (横長の行列では各列) の 2 乗の移動平均
                    rows, columns = grad.shape[0], grad[0].numel()
                    if not state:
                        state["momentum"] = torch.zeros_like(parameter)
                        state["step"]     = torch.zeros((), device=parameter.device)
                        if group["normuon"]:
                            state["square"] = torch.zeros((rows, 1) if rows >= columns else (1, columns), device=parameter.device)
                    
                    # モーメンタムを動かし (momentum_warmup_steps の間に momentum_start から momentum へ上げる)、Nesterov のステップを取る
                    state["step"] += 1
                    momentum       = group["momentum"] if group["momentum_warmup_steps"] == 0 else group["momentum_start"] + (group["momentum"] - group["momentum_start"]) * (state["step"] / group["momentum_warmup_steps"]).clamp(max=1.0)
                    state["momentum"].lerp_(grad, 1 - momentum)
                    update         = grad.lerp(state["momentum"], momentum).reshape(rows, columns)
                    
                    # 更新を一番近い直交行列に置き換える (bfloat16 で、短い辺を先にして計算する)
                    coefficients, safety = ORTHOGONALIZATIONS[group["orthogonalization"]]
                    x                    = update.bfloat16().mT if rows > columns else update.bfloat16()
                    x                    = x / (x.norm() * (1 + safety) + 1e-7)
                    for a, b, c in coefficients:
                        gram = x @ x.mT
                        x    = a * x + (b * gram + c * gram @ gram) @ x
                    update               = (x.mT if rows > columns else x).float()
                    
                    # NorMuon: 各行 (または各列) を 2 乗の移動平均の平方根で割り、更新全体の大きさは保つ
                    if group["normuon"]:
                        size   = update.norm()
                        state["square"].lerp_(update.square().mean(dim=1 if rows >= columns else 0, keepdim=True), 1 - group["beta2"])
                        update = update * state["square"].clamp_min(1e-10).rsqrt()
                        update = update * (size / update.norm().clamp_min(1e-10))
                    
                    # 形に合わせて更新をスケールし、weight decay を作る (cautious のときは、更新と重みの符号が同じところだけ)
                    update = update.reshape(parameter.shape) * max(1.0, rows / columns) ** 0.5
                    decay  = group["weight_decay"] * parameter * ((update * parameter) >= 0) if group["cautious"] else group["weight_decay"] * parameter
                else:
                    # 最初のステップで状態を作る: 勾配とその 2 乗の移動平均と、ステップ数
                    if not state:
                        state["average"] = torch.zeros_like(parameter)
                        state["square"]  = torch.zeros_like(parameter)
                        state["step"]    = torch.zeros((), device=parameter.device)
                    
                    # 移動平均を動かしてバイアス補正で割り、weight decay を作る
                    beta1, beta2   = group["betas"]
                    state["step"] += 1
                    state["average"].lerp_(grad, 1 - beta1)
                    state["square"].lerp_(grad.square(), 1 - beta2)
                    update         = (state["average"] / (1 - beta1 ** state["step"])) / ((state["square"] / (1 - beta2 ** state["step"])).sqrt() + group["eps"])
                    decay          = group["weight_decay"] * parameter
                
                # 学習率の分だけパラメータを動かす
                parameter.sub_(group["lr"] * (update + decay))
        return loss
