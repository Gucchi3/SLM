# @file bin2dataset.py
# @brief トークン化した .bin ファイルからトークンの窓を切り出すデータセット

from pathlib          import Path
from torch.utils.data import Dataset

import numpy as np
import torch



class Bin2dataset(Dataset):
    '''
    #* uint16 のトークン ID の .bin ファイルから max_length + 1 トークンの窓を切り出し、input_ids、attention_mask、targets を渡す。
    #? pretrain.py の build_pretrain_bundle() が使う。.bin ファイルは python -m tool.data で作る。
    #! repeat は窓を何回渡すかを決める。1 なら各窓を 1 回、2 なら 2 回、0.5 なら前半だけ。空のファイル (test データの無い test.bin など) には窓が無い。
    #! ファイルは、各 DataLoader ワーカーで最初の __getitem__() のときに開く。DataLoader を作る前に __getitem__() を呼ばないこと。呼ぶと Windows では各ワーカーがファイル全体のコピーを持ってしまう。
    '''
    
    ids: np.memmap | None
    
    def __init__(self, path: Path, max_length: int, repeat: float) -> None:
        '''
        #* パスと窓の長さを覚え、ファイルの大きさから窓の数を数える。
        '''
        
        super().__init__()
        self.path       = path
        self.max_length = max_length
        self.windows    = max((path.stat().st_size // 2 - 1) // max_length, 0)
        self.length     = round(self.windows * repeat)
        self.ids        = None
        return None
    
    def __len__(self) -> int:
        '''
        #* 窓の数 x repeat を返す。窓は max_length トークンずつずれるので、各トークンは 1 回の repeat の中で 1 回だけ正解になる。
        '''
        
        return self.length
    
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        '''
        #* 1 つの窓の input_ids、attention_mask、targets を返す。targets は input_ids を 1 トークンずらしたもの。
        '''
        
        # 最初の呼び出しのときに、このワーカーでファイルを開く
        if self.ids is None:
            self.ids = np.memmap(self.path, dtype=np.uint16, mode="r")
        
        # max_length + 1 トークンを切り出し、正解を 1 トークンずらす (最後の窓を超えた番号は最初の窓に戻る)
        start  = (index % self.windows) * self.max_length
        window = torch.from_numpy(self.ids[start:start + self.max_length + 1].astype(np.int64))
        return {"input_ids": window[:-1], "attention_mask": torch.ones(self.max_length, dtype=torch.int64), "targets": window[1:]}
