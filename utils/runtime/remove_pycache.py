# @file remove_pycache.py
# @brief Python のキャッシュを消す

from pathlib import Path

import shutil



def remove_pycache() -> None:
    '''
    #* 今のフォルダの下の __pycache__ フォルダをすべて消す。
    #? utils/trainer/pipeline.py と tool/ のスクリプトから呼ぶ。
    '''
    
    # 深いフォルダから先に消す
    cache_dirs = sorted(Path.cwd().rglob("__pycache__"), key=lambda path: len(path.parts), reverse=True)
    for cache_dir in cache_dirs:
        if cache_dir.is_dir() and not cache_dir.is_symlink():
            shutil.rmtree(cache_dir)
    return None
