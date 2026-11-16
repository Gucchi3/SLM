# @file bundle.py
# @brief データセットの名前から DataBundle を作る関数を選ぶ

##########################################################
'''
データセットの追加
1. utils/data/xxx.py に、DataBundle を返す build_xxx_bundle(config, seed) を書く。
2. DATASET_REGISTRY に "名前: build_xxx_bundle" を足す。
3. config.data.name をその名前にする。
'''
##########################################################

from collections.abc import Callable
from types           import SimpleNamespace

from utils.exceptions import Error
from .pretrain        import build_pretrain_bundle
from .sft             import build_sft_bundle
from .types           import DataBundle

DatasetBuilder = Callable[[SimpleNamespace, int], DataBundle]

DATASET_REGISTRY: dict[str, DatasetBuilder] = {
    "pretrain": build_pretrain_bundle,
    "sft"     : build_sft_bundle,
}



def build_data_bundle(config: SimpleNamespace, seed: int) -> DataBundle:
    '''
    #* config.name の DataBundle を返す。
    #? utils/trainer/pipeline.py から config.data を渡して呼ぶ。
    '''
    
    if config.name not in DATASET_REGISTRY:
        raise Error("Unknown dataset.", detail=f"data.name: {config.name!r}\nAvailable: {', '.join(DATASET_REGISTRY)}", hint="Add a builder to DATASET_REGISTRY in utils/data/bundle.py.", code=5)
    return DATASET_REGISTRY[config.name](config, seed)
