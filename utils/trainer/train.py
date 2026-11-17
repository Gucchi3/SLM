# @file train.py
# @brief 決まった回数だけ重みを更新してモデルを学習する

from torch.nn.utils   import clip_grad_norm_
from torch.utils.data import DataLoader
from types            import SimpleNamespace

import random
import torch
import torch.nn as nn

from utils.data                 import EpochSampler
from utils.runtime              import print_progress
from utils.trainer.optimization import loss_distill, loss_label
from .types                     import Metrics, TrainState, TrainingComponents

GRAPH_WARMUP_UPDATES = 3
# train.compile のときに compile した、正解との損失と蒸留の損失 (torch.compile は最初の呼び出しまで何もしない)
#! 小さなモデルでは、時間の大半がここの語彙全体にわたる要素ごとの計算 (softmax、z-loss) にかかり、compile がそれをまとめる。
LOSS_LABEL_COMPILED   = torch.compile(loss_label)
LOSS_DISTILL_COMPILED = torch.compile(loss_distill)



def _train_forward_backward(config: SimpleNamespace, model: nn.Module, teacher: nn.Module | None, components: TrainingComponents, inputs: dict[str, torch.Tensor], totals: dict[str, torch.Tensor], times: int | None) -> None:
    '''
    #* micro-batch 1 つの forward と backward を行い、正解との損失と正解率を合計に足す。
    #? train_steps() から、そのままでも CUDA グラフのキャプチャの中でも呼ぶ。
    #! ここでテンソルの値を CPU で読まないこと。CUDA グラフはそれを持てない。
    #! times はループの周回数 (hybrid の loop.times_train から選んだもの)。None なら model に渡さず、モデルの決まった回数で通す。
    '''
    
    # 生徒を通す (CUDA では BF16 の AMP。CUDA グラフのためにキャストのキャッシュは使わない)
    device_type = inputs["input_ids"].device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda", cache_enabled=False):
        logits = model(inputs["input_ids"], inputs["attention_mask"]) if times is None else model(inputs["input_ids"], inputs["attention_mask"], times=times)
    
    # 正解との損失を計算する
    label_fn                   = LOSS_LABEL_COMPILED if config.train.compile and device_type == "cuda" else loss_label
    label_loss, correct, count = label_fn(logits, inputs["targets"], components.loss_fn, components.loss_kind)
    loss                       = label_loss
    
    # 温度でやわらげた教師の出力との損失を混ぜる
    if teacher is not None:
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda", cache_enabled=False):
            teacher_logits = teacher(inputs.get("teacher_input_ids", inputs["input_ids"]), inputs.get("teacher_attention_mask", inputs["attention_mask"]))
        temperature = config.train.distillation.temperature
        alpha       = config.train.distillation.alpha
        distill_fn  = LOSS_DISTILL_COMPILED if config.train.compile and device_type == "cuda" else loss_distill
        soft_loss   = distill_fn(logits, teacher_logits, inputs["targets"], components.loss_kind, temperature)
        loss        = (1 - alpha) * label_loss + alpha * temperature**2 * soft_loss
    
    # 勾配を足す。micro-batch 1 周分でその平均になるようにスケールする
    (loss / config.train.accumulation_steps).backward()
    
    # 正解との損失 (正解の数で重みを付ける) と正解率を合計に足す
    totals["loss"].add_(label_loss.detach().double() * count)
    totals["correct"].add_(correct)
    totals["counted"].add_(count)
    return None


def _train_update(config: SimpleNamespace, model: nn.Module, components: TrainingComponents) -> None:
    '''
    #* 勾配をクリップして重みを更新し、勾配を 0 にする。
    #? train_steps() から、そのままでも CUDA グラフのキャプチャの中でも呼ぶ。
    #! 勾配のテンソルは残す (set_to_none=False)。CUDA グラフは同じメモリに足していくため。スケジューラーはここに移さないこと。CUDA グラフが同じ学習率を再生してしまう。
    '''
    
    clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
    components.optimizer.step()
    components.optimizer.zero_grad(set_to_none=False)
    return None


def train_steps(config: SimpleNamespace, device: torch.device, model: nn.Module, teacher: nn.Module | None, train_loader: DataLoader, train_sampler: EpochSampler, components: TrainingComponents, state: TrainState, updates: int) -> Metrics:
    '''
    #* 決まった回数だけ重みを更新してモデルを学習する。勾配の蓄積とクリップを行い、train データが終われば次のエポックへ進む。教師があれば、温度でやわらげた教師の出力も学ぶ (蒸留)。
    #? 検証の合間に loop.py から呼ぶ。学習全体の進み具合と途中の損失を 1 行で表示し、終わると消す。
    #! train データはエポック state.epoch の micro-batch state.batch から始まり (train_sampler を通して)、state.step、state.epoch、state.batch は学習とともに進む。
    #! 各バッチは input_ids、attention_mask、targets (飛ばすところは -100) の辞書。教師が別のトークナイザーを使うときは、teacher_input_ids と teacher_attention_mask を足す。
    #! ロジットと正解の形は損失の種類に従う (optimization/loss.py の loss_label() を参照)。教師は生徒と同じ形を返すこと。
    #! CUDA で train.compile のときは、model は pipeline.py で compile したモデルで、損失も compile して動く。compile は CUDA グラフのキャプチャの前、最初のそのままの更新の間に起きる。mode="reduce-overhead" は使わないこと。自分で CUDA グラフを作ってしまう。
    #! CUDA で train.cuda_graph のときは、各呼び出しの最初の数回の更新はそのまま動く。その後は、micro-batch 1 つと更新 1 回をそれぞれ CUDA グラフで動かす。形の違うバッチはそのまま動く。
    #! モデルが loop_times_train (hybrid の loop.times_train) を持ち空でなければ、重みの更新ごとにその候補から周回数を選ぶ。乱数は run.seed と更新の番号だけで決まるので、再開しても複数の GPU でも同じ数になる。
    #! 候補が 2 種類以上のときは CUDA Graph を使えない (pipeline.py で止める)。1 種類なら、その回数のままグラフにする。
    '''
    
    # 学習モードにし、デバイス上の合計を 0 に戻す
    model.train()
    accumulation_steps = config.train.accumulation_steps
    use_graph          = config.train.cuda_graph and device.type == "cuda"
    side_stream        = torch.cuda.Stream() if use_graph else None
    totals             = {"loss": torch.zeros((), device=device, dtype=torch.float64), "correct": torch.zeros((), device=device, dtype=torch.int64), "counted": torch.zeros((), device=device, dtype=torch.int64)}
    static             = dict[str, torch.Tensor]()
    choices            = tuple(getattr(model, "loop_times_train", ()))
    times              = None
    forward_graph      = None
    update_graph       = None
    samples            = 0
    micro_batches      = 0
    done               = 0
    
    # CUDA グラフのために、横のストリームで動かす (PyTorch がウォームアップとキャプチャにそう求めるため)
    if side_stream is not None:
        side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream), print_progress("train") as progress:
        task = progress.add_task("Train", total=config.train.steps, completed=state.step, loss="-")
        while done < updates:
            # 学習が進んだ位置から、エポックの train データを始める
            train_sampler.set_position(state.epoch, state.batch * config.data.batch_size)
            for batch in train_loader:
                micro_batches += 1
                state.batch   += 1
                
                # 形が合えばバッチを CUDA グラフの固定のバッファにコピーし、合わなければデバイスへ送る
                if use_graph and not static:
                    static = {key: value.to(device) for key, value in batch.items()}
                same_shape = use_graph and all(static[key].shape == value.shape for key, value in batch.items())
                if same_shape:
                    for key, value in batch.items():
                        static[key].copy_(value, non_blocking=True)
                inputs     = static if same_shape else {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                
                # 更新の始めに、その更新で使うループの周回数を選ぶ
                if choices and (micro_batches - 1) % accumulation_steps == 0:
                    times = random.Random(config.run.seed * 1_000_003 + state.step).choice(choices)
                
                # ウォームアップの更新の後、1 周の始めに CUDA グラフをキャプチャする
                if same_shape and forward_graph is None and micro_batches > GRAPH_WARMUP_UPDATES * accumulation_steps and (micro_batches - 1) % accumulation_steps == 0:
                    forward_graph = torch.cuda.CUDAGraph()
                    update_graph  = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(forward_graph, capture_error_mode="thread_local"):
                        _train_forward_backward(config, model, teacher, components, static, totals, times)
                    with torch.cuda.graph(update_graph, pool=forward_graph.pool(), capture_error_mode="thread_local"):
                        _train_update(config, model, components)
                
                # micro-batch を CUDA グラフかそのままで動かす
                if forward_graph is not None and same_shape:
                    forward_graph.replay()
                else:
                    _train_forward_backward(config, model, teacher, components, inputs, totals, times)
                samples += inputs["input_ids"].shape[0]
                
                # 1 周の終わりに重みを更新し、学習率を 1 ステップ進める
                if micro_batches % accumulation_steps == 0 and update_graph is not None:
                    update_graph.replay()
                elif micro_batches % accumulation_steps == 0:
                    _train_update(config, model, components)
                if micro_batches % accumulation_steps == 0:
                    components.scheduler.step()
                    done       += 1
                    state.step += 1
                    
                    # log_interval 回の更新ごとに途中の損失を表示する (GPU を待つ)
                    progress.update(task, advance=1)
                    if state.step % config.train.log_interval == 0:
                        progress.update(task, loss=f"{(totals['loss'] / totals['counted'].clamp(min=1)).item():.4f}")
                if done == updates:
                    break
            else:
                # train データが終わったら次のエポックへ進む
                state.epoch += 1
                state.batch  = 0
    if side_stream is not None:
        torch.cuda.current_stream().wait_stream(side_stream)
    
    # 最後に 1 回だけ合計を読む
    counted = max(int(totals["counted"].item()), 1)
    return Metrics(totals["loss"].item() / counted, totals["correct"].item() / counted, samples)
