# @file exceptions.py
# @brief 詳細付きのエラーのクラス

from rich.console import Console
from rich.text    import Text



class Error(SystemExit):
    '''
    #* エラーのメッセージを詳細とヒント付きで表示し、エラー番号でプログラムを止める。
    #! エラーごとに別の番号を付ける。同じ番号を 2 回使わない。
    #! detail、hint、code はキーワード引数で渡す (例: Error("Invalid device.", detail=f"run.device: {device!r}", hint="Use auto, cpu, or cuda.", code=4))。
    '''
    
    def __init__(self, message: str, *, detail: str | None = None, hint: str | None = None, code: int) -> None:
        '''
        #* 詳細を覚え、トレースバック無しで stderr に表示する。
        '''
        
        # 詳細と終了コードを覚える
        super().__init__(code)
        self.message = message
        self.detail  = detail
        self.hint    = hint
        
        # エラーを stderr に表示する
        console = Console(stderr=True)
        console.print(Text(f"[ERROR] {message}", style="bold red"))
        if detail is not None:
            console.print(Text(detail))
        if hint is not None:
            console.print(Text(f"Hint: {hint}"))
        return None
    
    @staticmethod
    def show_interrupted() -> None:
        '''
        #* Ctrl+C が押されたときに、stderr にメッセージを表示する。
        #? main.py から呼ぶ。
        '''
        
        Console(stderr=True).print(Text("[Stopped] The user stopped the program."))
        return None
