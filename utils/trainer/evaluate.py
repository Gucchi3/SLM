# @file evaluate.py
# @brief ローダーの全バッチでモデルを評価する

from torch.utils.data import DataLoader
from types            import SimpleNamespace

import torch
import torch.nn as nn

from utils.runtime              import print_progress
from utils.trainer.optimization import loss_label
from .types                     import Metrics, TrainingComponents



def evaluate_epoch(config: SimpleNamespace, device: torch.device, model: nn.Module, loader: DataLoader, components: TrainingComponents, description: str, low_precision: bool, times: int | None = None) -> Metrics:
    '''
    #* 学習はせずに、ローダーでのモデルの損失と正解率を求める。
    #? 検証では loop.py から (low_precision=True、速い BF16)、テストでは pipeline.py から (low_precision=False、正確な float32) 呼ぶ。
    #! バッチのキーは train.py の train_steps() と同じ。損失は、飛ばさない正解についての平均。
    #! times はループの周回数で、None なら model に渡さない (hybrid は loop.times で通す)。
    '''
    
    # 評価モードにし、デバイス上の合計を 0 に戻す
    model.eval()
    amp_enabled = low_precision and device.type == "cuda"
    loss_total  = torch.zeros((), device=device, dtype=torch.float64)
    correct     = torch.zeros((), device=device, dtype=torch.int64  )
    counted     = torch.zeros((), device=device, dtype=torch.int64  )
    samples     = 0
    
    # 勾配なしで各バッチを評価する
    with torch.no_grad(), print_progress("validation") as progress:
        task = progress.add_task(description, total=len(loader), loss="-")
        for step, batch in enumerate(loader, start=1):
            # モデルを通す (low_precision のときは CUDA で BF16 の AMP)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                mask   = batch["attention_mask"].to(device, non_blocking=True)
                logits = model(input_ids, mask) if times is None else model(input_ids, mask, times=times)
            
            # 正解との損失を計算する
            loss, batch_correct, count = loss_label(logits, batch["targets"].to(device, non_blocking=True), components.loss_fn, components.loss_kind)
            
            # バッチの結果を合計に足す (損失は正解の数で重みを付ける)
            loss_total.add_(loss.double() * count)
            correct.add_(batch_correct)
            counted.add_(count)
            samples += input_ids.shape[0]
            
            # log_interval ステップごとに途中の損失を表示する (GPU を待つ)
            progress.update(task, advance=1)
            if step % config.train.log_interval == 0:
                progress.update(task, loss=f"{(loss_total / counted.clamp(min=1)).item():.4f}")
    
    # 最後に 1 回だけ合計を読む
    total = max(int(counted.item()), 1)
    return Metrics(loss_total.item() / total, correct.item() / total, samples)
