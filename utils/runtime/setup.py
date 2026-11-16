# @file setup.py
# @brief 実行の準備

from datetime import datetime
from pathlib  import Path

import numpy as np
import random
import torch

from utils.exceptions import Error



def setup_seed(seed: int) -> None:
    '''
    #* Python、NumPy、PyTorch のシードを設定する。
    '''
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return None


def setup_device(device: str) -> torch.device:
    '''
    #* run.device から CPU か CUDA のデバイスを選ぶ。
    #! run.device は auto、cpu、cuda のどれか。auto は PyTorch が CUDA を使えれば CUDA を選ぶ。
    '''
    
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "cuda":
        raise Error("CUDA is not available.", detail="run.device is 'cuda', but PyTorch cannot find a CUDA device.", hint="Check CUDA, or set run.device to 'auto' or 'cpu'.", code=3)
    raise Error("Invalid device.", detail=f"run.device: {device!r}", hint="Set run.device to 'auto', 'cuda', or 'cpu'.", code=4)


def setup_log_dir(log_dir: str | Path) -> Path:
    '''
    #* log_dir の中に日付と時刻の名前の新しいフォルダを作り、それを返す。
    #! 同じ時刻のフォルダがあれば、_01 のような番号を付ける。
    '''
    
    root  = Path(log_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        root.mkdir(parents=True, exist_ok=True)
        index = 0
        while True:
            candidate = root / (stamp if index == 0 else f"{stamp}_{index:02d}")
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                index += 1
    except OSError as error:
        raise Error("Could not make the log folder.", detail=f"{root}: {error}", hint="Check run.log_dir and the write permission.", code=7)
