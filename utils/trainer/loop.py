# @file loop.py
# @brief 学習ループ: 最後のステップまで、学習、検証、保存をくり返す

from dataclasses import asdict
from pathlib     import Path
from types       import SimpleNamespace

import torch
import torch.nn as nn

from utils.data    import DataBundle
from utils.runtime import print_validation, save_graph, save_history
from .evaluate     import evaluate_epoch
from .train        import train_steps
from .types        import TrainState, TrainingComponents



def loop_train(config: SimpleNamespace, device: torch.device, log_dir: Path, bundle: DataBundle, model: nn.Module, train_model: nn.Module, teacher: nn.Module | None, components: TrainingComponents, state: TrainState) -> None:
    '''
    #* 重みの更新が train.steps 回になるまで学習する。train.eval_interval 回ごとと最後に、検証して、重み、履歴、グラフ、学習の状態を保存する。
    #? pipeline.py から呼ぶ。train.py の train_steps() と evaluate.py の evaluate_epoch() を使う。
    #! state はどこから始めるかを表す: 最初のステップか、train.resume の train_state.pt に保存した位置。再開した学習は最後の検証の位置からやり直す。
    #! best.pth には検証の perplexity が一番低い重みを残す。教師は蒸留のときだけ渡す。
    #! train_model は model そのものか、train.compile で compile した model (重みは同じ)。使うのは学習だけ。検証と保存は model を使うので、重みの名前に compile の接頭辞が付かず、形の違う最後のバッチで compile し直すこともない。
    #! 検証の前の empty_cache() は、GPU メモリ (と Windows のメモリのコミット) の最大値を下げる。結果は変わらない。
    '''
    
    # 最後のステップまで、学習と検証を交互に行う
    while state.step < config.train.steps:
        # 次の eval_interval の倍数か最後のステップまで学習し、検証する
        updates       = min(config.train.eval_interval - state.step % config.train.eval_interval, config.train.steps - state.step)
        train_metrics = train_steps(config, device, train_model, teacher, bundle.train_loader, bundle.train_sampler, components, state, updates)
        # train_steps() が手放した CUDA グラフのメモリを返す。もう使わず、次のキャプチャまで残ったままになるため
        torch.cuda.empty_cache()
        val_metrics   = evaluate_epoch(config, device, model, bundle.validation_loader, components, "Validation", low_precision=True)
        
        # 学習で周回数をランダムに選ぶときは、候補の周回数ごとの PPL も測る
        times_ppl = {f"val_ppl_x{times}": evaluate_epoch(config, device, model, bundle.validation_loader, components, f"Validation x{times}", low_precision=True, times=times).perplexity for times in sorted(set(getattr(model, "loop_times_train", ())))}
        
        # 最新の重みと、検証の perplexity で最良の重みを保存する
        torch.save(model.state_dict(), log_dir / "latest.pth")
        if val_metrics.perplexity < state.best_ppl:
            state.best_ppl = val_metrics.perplexity
            torch.save(model.state_dict(), log_dir / "best.pth")
        
        # 結果を記録、保存、表示する (tokens は、ここまでの全部の重みの更新の正解トークンの数)
        record = {
            "step"         : state.step,
            "tokens"       : state.step * config.data.batch_size * config.train.accumulation_steps * config.data.max_length,
            "learning_rate": float(components.optimizer.param_groups[0]["lr"]),
            "train_loss"   : train_metrics.loss,
            "val_loss"     : val_metrics.loss,
            "val_ppl"      : val_metrics.perplexity,
            "val_acc"      : val_metrics.accuracy,
            "best_ppl"     : state.best_ppl,
            **times_ppl,
        }
        state.history.append(record)
        save_history(log_dir, state.history)
        save_graph(state.history, log_dir)
        print_validation(state.history, config.train.steps)
        
        # 学習の状態を保存し、train.resume でここから再開できるようにする
        torch.save({"model": model.state_dict(), "optimizer": components.optimizer.state_dict(), "scheduler": components.scheduler.state_dict(), "state": asdict(state)}, log_dir / "train_state.pt")
    
    return None
