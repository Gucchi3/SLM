# @file epoch_sampler.py
# @brief 学習データをエポックごとに新しいランダムな順で渡し、エポックの途中からも始められるサンプラー

from collections.abc  import Iterator
from torch.utils.data import Sampler

import numpy as np



class EpochSampler(Sampler[int]):
    '''
    #* 0 から length - 1 の番号を、シードとエポックだけで決まるランダムな順に、その順の決めた位置から渡す。
    #? pretrain.py の build_pretrain_bundle() が train のローダーに使う。utils/trainer/train.py の train_steps() が set_position() で位置を決める。
    #! エポックの順はシードとエポックの番号から作り直す。だから train_state.pt から再開した学習でも、止めなかった学習と同じバッチになる。
    '''
    
    def __init__(self, length: int, seed: int) -> None:
        '''
        #* 番号の数とシードを覚え、エポック 0 の最初から始める。
        '''
        
        super().__init__()
        self.length = length
        self.seed   = seed
        self.epoch  = 0
        self.start  = 0
        return None
    
    def set_position(self, epoch: int, start: int) -> None:
        '''
        #* 次の 1 周で、そのエポックの順を start 番目から渡すようにする。
        '''
        
        self.epoch = epoch
        self.start = start
        return None
    
    def __len__(self) -> int:
        '''
        #* 次の 1 周で渡す番号の数を返す。
        '''
        
        return self.length - self.start
    
    def __iter__(self) -> Iterator[int]:
        '''
        #* エポックの番号をランダムな順に、start 番目から渡す。
        #! 番号は Python のリストではなく NumPy の配列から取り出す。数千万個の番号のリストは 1 GB を超えるため。
        '''
        
        order = np.random.default_rng([self.seed, self.epoch]).permutation(self.length)
        return iter(order[self.start:])
