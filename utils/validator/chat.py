# @file chat.py
# @brief 学習したモデルとターミナルで会話する

from rich.console import Console
from rich.text    import Text
from types        import SimpleNamespace

from utils.exceptions import Error
from .generate        import generate_tokens
from .load            import load_model

CHAT_STYLES = ("text", "chat")



def chat_model(config: SimpleNamespace) -> None:
    '''
    #* モデル、その重み、トークナイザーを読み込み、ターミナルでモデルと会話する。
    #? main.py で run.mode (または --mode) が chat のとき選ばれる。load.py の load_model() と generate.py の generate_tokens() を使う。
    #! validator.chat.style が text: 入力した文の続きをモデルが書く (事前学習だけのモデル向け)。chat: 入力した文を <user> 文 <bot> として入れ、モデルは <eos> まで答える (SFT 後のモデル向け)。
    #! chat の形では、前の発話を文脈に残し、モデルは最後の data.max_length トークンを見る。/reset で発話を忘れ、/exit か空行で終わる。
    '''
    
    # 形を確かめる
    settings = config.validator.chat
    if settings.style not in CHAT_STYLES:
        raise Error("Unknown chat style.", detail=f"validator.chat.style: {settings.style!r}\nAvailable: {', '.join(CHAT_STYLES)}", hint="Use text for a model after pretraining, and chat for a model after SFT.", code=32)
    
    # モデルとトークナイザーを読み込み、chat の形ではトークナイザーに発話のトークンがあることを確かめる
    model, tokenizer, device = load_model(config)
    vocab                    = tokenizer.get_vocab()
    if settings.style == "chat" and not {"<user>", "<bot>", "<eos>"} <= vocab.keys():
        raise Error("The tokenizer has no tokens for the turns.", detail=f"{config.data.tokenizer_dir} has no <user>, <bot>, or <eos>.", hint="Use validator.chat.style text, or a tokenizer made by tool/data.", code=34)
    
    # /exit か空行まで会話する
    console = Console()
    history = list[int]()
    console.print(Text(f"[Chat] Style: {settings.style}. Type /reset to forget the talk, and /exit or an empty line to end.", style="cyan"))
    while True:
        try:
            text = console.input("[bold cyan]You[/bold cyan]: ").strip()
        except EOFError:
            break
        if text in ("", "/exit"):
            break
        if text == "/reset":
            history.clear()
            continue
        
        # 答えを作る。chat の形では、発話を前の発話の後ろに置き、答えと <eos> を次の発話のために残す
        if settings.style == "chat":
            prompt = history + [vocab["<user>"]] + tokenizer.encode(text).ids + [vocab["<bot>"]]
        else:
            # prompt = tokenizer.encode(text).ids
            prompt = [vocab["<eos>"]] + tokenizer.encode(text).ids
        answer = generate_tokens(model, prompt, device, config.data.max_length, settings.max_new_tokens, settings.temperature, settings.top_k, vocab.get("<eos>"))
        if settings.style == "chat":
            history = prompt + answer + [vocab["<eos>"]]
        console.print(Text("Model: ", style="bold green") + Text(tokenizer.decode(answer)))
    return None
