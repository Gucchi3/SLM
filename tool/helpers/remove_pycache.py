# @file remove_pycache.py
# @brief コマンドラインから Python のキャッシュを消す

from utils.runtime import remove_pycache



def run_remove_pycache() -> None:
    '''
    #* 今のフォルダの下の __pycache__ フォルダをすべて消す。
    #! プロジェクトのフォルダから実行する (例: python -m tool.helpers.remove_pycache)。
    '''
    
    remove_pycache()
    return None


if __name__ == "__main__":
    run_remove_pycache()
