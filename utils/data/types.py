# @file types.py
# @brief データの型

from dataclasses      import dataclass
from math             import prod
from torch.utils.data import DataLoader
from typing           import Any

from .epoch_sampler import EpochSampler

'''
DataBundle
├── train_loader
├── train_sampler: EpochSampler
├── validation_loader
├── test_loader
└── metadata: DatasetMetadata
    ├── name
    ├── input_shape / input_size
    ├── num_classes / class_names
    ├── train_size / validation_size / test_size
    ├── split_seed
    └── preprocessing: PreprocessingMetadata
        ├── name
        ├── parameters
        └── output_dtype
'''



@dataclass(frozen=True)
class PreprocessingMetadata:
    '''
    #* トークナイザーなど、データの前処理の詳細を持つ。
    '''
    
    name        : str
    parameters  : dict[str, str | int | float | bool]
    output_dtype: str


@dataclass(frozen=True)
class DatasetMetadata:
    '''
    #* データセットとその分割の詳細を持つ。
    #? utils/runtime/save.py の save_metadata() が metadata.json に保存する。
    '''
    
    name           : str
    input_shape    : tuple[int, ...] | None
    num_classes    : int | None
    class_names    : tuple[str, ...] | None
    train_size     : int
    validation_size: int
    test_size      : int
    split_seed     : int
    preprocessing  : PreprocessingMetadata
    
    @property
    def input_size(self) -> int | None:
        '''
        #* 形が決まった入力 1 つの要素数を返す。
        #! 長さがばらばらの入力では None を返す。
        '''
        
        return prod(self.input_shape) if self.input_shape is not None else None


@dataclass(frozen=True)
class DataBundle:
    '''
    #* train、validation、test のローダーを、train のローダーのサンプラーとメタデータと一緒にまとめる。
    #! 学習のコードは train_sampler を学習の位置 (再開した学習の位置など) に動かす。
    '''
    
    train_loader     : DataLoader[Any]
    train_sampler    : EpochSampler
    validation_loader: DataLoader[Any]
    test_loader      : DataLoader[Any]
    metadata         : DatasetMetadata
