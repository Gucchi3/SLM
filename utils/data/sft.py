# @file sft.py
# @brief 日常会話の SFT 用の DataBundle を作る (今は空)

from types import SimpleNamespace

from .types import DataBundle



def build_sft_bundle(config: SimpleNamespace, seed: int) -> DataBundle:
    '''
    #* SFT (会話のボットの返事で次のトークンを当てる) 用の train、validation、test のローダーを返す。
    #? data.name が "sft" のとき bundle.py で選ばれる。bin2dataset.py の Bin2dataset を使う。
    #! まだ書いていない。正解にするのはボットの返事だけで、ユーザーの発話とパディングは -100 にする。
    '''
    
    raise NotImplementedError("build_sft_bundle() is not written yet (utils/data/sft.py).")
