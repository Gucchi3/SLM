# @file save.py
# @brief 学習のファイルを保存する

from dataclasses       import asdict
from matplotlib.figure import Figure
from pathlib           import Path
from types             import SimpleNamespace

import json

from utils.data import DatasetMetadata
from .print     import LOSS_COLORS



def save_metadata(metadata: DatasetMetadata, config: SimpleNamespace, log_dir: Path) -> None:
    '''
    #* データセットの詳細と config 全体を metadata.json に保存する。
    #? utils/trainer/pipeline.py から呼ぶ。
    '''
    
    metadata_data           = asdict(metadata)
    metadata_data["config"] = config
    (log_dir / "metadata.json").write_text(json.dumps(metadata_data, default=vars, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return None


def save_history(log_dir: Path, history: list[dict[str, int | float]]) -> None:
    '''
    #* 学習の履歴を result.json に保存する。
    #? 検証のたびに utils/trainer/loop.py から呼ぶ。
    '''
    
    (log_dir / "result.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return None


def save_graph(history: list[dict[str, int | float]], log_dir: Path) -> None:
    '''
    #* 損失と正解率の曲線を、重みの更新回数を横軸にして training_curves.png に保存する。
    #? 検証のたびに utils/trainer/loop.py から呼ぶ。
    #! 図は pyplot ではなく Figure で作るので、画面の仕組み (Windows の Tk) を使わない。Tk の図が別のスレッド (DataLoader の pin_memory のスレッドなど) で片付けられると、Tcl_AsyncDelete で学習全体が止まる。
    '''
    
    # 履歴から各値を取り出す
    steps        = [record["step"]       for record in history]
    train_losses = [record["train_loss"] for record in history]
    val_losses   = [record["val_loss"]   for record in history]
    val_accs     = [record["val_acc"]    for record in history]
    
    # 検証の曲線の凡例を、最良と最新の値付きで作る
    loss_label = f"Validation loss (ppl best: {history[-1]['best_ppl']:.2f}, latest: {history[-1]['val_ppl']:.2f})"
    acc_label  = f"Validation accuracy (best: {max(val_accs):.3f}, latest: {val_accs[-1]:.3f})"
    
    # グラフ 2 つの図を作る
    fig     = Figure(figsize=(12, 4), layout="constrained")
    ax_loss = fig.add_subplot(1, 2, 1)
    ax_acc  = fig.add_subplot(1, 2, 2)
    
    # 左: train の損失を水色、validation の損失をオレンジで描く
    ax_loss.plot(steps, train_losses, color=LOSS_COLORS["train"]     , marker=None, label="Train loss")
    ax_loss.plot(steps, val_losses  , color=LOSS_COLORS["validation"], marker=None, label=loss_label  )
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="best")
    
    # 右: validation の正解率
    ax_acc.plot(steps, val_accs, color="red", marker=None, label=acc_label)
    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(loc="best")
    
    # 図を保存する
    fig.savefig(log_dir / "training_curves.png", dpi=150)
    return None
