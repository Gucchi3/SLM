# @file pipeline.py
# @brief 学習の流れ

from pathlib      import Path
from rich.console import Console
from rich.panel   import Panel
from tokenizers   import Tokenizer
from types        import SimpleNamespace

import torch

from model                      import build_model
from utils.data                 import build_data_bundle
from utils.exceptions           import Error
from utils.runtime              import print_metadata, remove_pycache, save_metadata, setup_device, setup_log_dir, setup_seed
from utils.trainer.optimization import build_loss_fn, build_optimizer, build_scheduler
from utils.validator            import benchmark_check, benchmark_tasks
from .evaluate                  import evaluate_epoch
from .loop                      import loop_train
from .types                     import TrainState, TrainingComponents

TRAIN_METHODS = ("normal", "distill")



def pipeline_train(config: SimpleNamespace) -> None:
    '''
    #* データ、モデル、学習の部品を準備し、学習してからテストする。
    #? main.py から呼ぶ。loop.py の loop_train() を呼び、そこから train.py と evaluate.py を呼ぶ。
    #! train.method は normal か distill にする。distill では model.teacher_name、model.teacher_checkpoint、train.distillation も要る (教師の形は model.teacher_spec、null なら model.spec と同じ)。
    #! train.resume に train_state.pt を指定すると、同じログのフォルダで、その状態から学習を続ける。モデルと最適化器は保存した学習と同じにする。
    #! 学習の後のテストには best.pth (検証の perplexity が一番低い重み) を使う。
    #! test データが無いとき (tool/data の --split で test が 0) は、テストの代わりに validator.benchmark のベンチマークを best.pth で実行し、benchmark.json をログのフォルダに置く。
    '''
    
    # 学習の方法と学習の長さを確かめる
    if config.train.method not in TRAIN_METHODS:
        raise Error("Unknown training method.", detail=f"train.method: {config.train.method!r}\nAvailable: {', '.join(TRAIN_METHODS)}", hint="Set train.method to 'normal' or 'distill'.", code=9)
    if config.train.steps < 1 or config.train.eval_interval < 1:
        raise Error("Invalid length of the training.", detail=f"train.steps: {config.train.steps}, train.eval_interval: {config.train.eval_interval}", hint="Use 1 or more for both.", code=30)
    
    # 学習でループの周回数を 2 種類以上から選ぶときは、CUDA Graph (決まった計算をくり返す) を使えない
    loop    = getattr(getattr(config.model, "spec", None), "loop", None)
    choices = sorted(set(getattr(loop, "times_train", None) or []))
    if len(choices) >= 2 and config.train.cuda_graph:
        raise Error("CUDA Graph cannot change the number of loops.", detail=f"model.spec.loop.times_train: {choices}, train.cuda_graph: true", hint="Set train.cuda_graph to false, or use one number in model.spec.loop.times_train.", code=43)
    
    # シードとデバイスを設定する
    setup_seed(config.run.seed)
    device = setup_device(config.run.device)
    
    # データを準備し、train データでバッチが 1 つ埋まることを確かめる。test データが無いときは、学習の後にベンチマークを実行するので、ここでベンチマークの名前を確かめる
    bundle = build_data_bundle(config.data, config.run.seed)
    if len(bundle.train_loader) == 0:
        raise Error("The train data is smaller than one batch.", detail=f"data.batch_size: {config.data.batch_size}", hint="Use a smaller data.batch_size, or more train data.", code=31)
    if len(bundle.test_loader) == 0:
        benchmark_check(config)
    
    # 生徒と、蒸留用の固定した教師を準備する
    model   = build_model(config.model.name, config.model.checkpoint, getattr(config.model, "spec", None)).to(device=device, dtype=torch.float32)
    teacher = None
    if config.train.method == "distill":
        teacher_spec = getattr(config.model, "teacher_spec", None)
        teacher      = build_model(config.model.teacher_name, config.model.teacher_checkpoint, getattr(config.model, "spec", None) if teacher_spec is None else teacher_spec).to(device)
        teacher.eval()
        teacher.requires_grad_(False)
    
    # 生徒を学習のためだけに compile する (重みは共有するので、保存、読み込み、最適化器はこれまでどおり model を使う)
    train_model = torch.compile(model) if config.train.compile and device.type == "cuda" else model
    
    # 損失、最適化器、スケジューラーを作る (スケジューラーは重みを更新するたびに 1 ステップ進む)
    loss_fn, loss_kind = build_loss_fn(config)
    optimizer          = build_optimizer(config, model)
    scheduler          = build_scheduler(config, optimizer, config.train.steps)
    components         = TrainingComponents(loss_fn, loss_kind, optimizer, scheduler)
    
    # train.resume の学習の状態を読むか、新しいログのフォルダで最初のステップから始める
    state   = TrainState(step=0, epoch=0, batch=0, best_ppl=float("inf"), history=[])
    log_dir = setup_log_dir(config.run.log_dir) if config.train.resume is None else Path(config.train.resume).parent
    if config.train.resume is not None:
        try:
            saved = torch.load(config.train.resume, map_location=device, weights_only=True)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
        except (OSError, KeyError, RuntimeError, ValueError) as error:
            raise Error("Could not load the training state.", detail=f"{config.train.resume}: {error}", hint="Set train.resume to a train_state.pt of a run with the same model and optimizer, or to null.", code=29)
        state = TrainState(**saved["state"])
    
    # 設定を保存して表示する
    save_metadata(bundle.metadata, config, log_dir)
    print_metadata(config)
    remove_pycache()
    
    # 学習し、最良の重み (検証の perplexity が一番低いもの) を読み込む
    loop_train(config, device, log_dir, bundle, model, train_model, teacher, components, state)
    best_path = log_dir / "best.pth"
    if not best_path.exists():
        raise Error("There are no best weights.", detail=f"Missing: {best_path}", hint="The validation perplexity was never a finite number. Check the learning rate and the data, then train again.", code=37)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    
    # 最良の重みを float32 でテストする。test データが無いときはベンチマークを実行する
    if len(bundle.test_loader) > 0:
        test_metrics = evaluate_epoch(config, device, model, bundle.test_loader, components, "Test", low_precision=False)
        Console().print(Panel(f"[bold green]Test loss: {test_metrics.loss:.4f} | Test perplexity: {test_metrics.perplexity:.2f} | Test accuracy: {test_metrics.accuracy:.2%}[/bold green]", border_style="green", expand=False))
    else:
        benchmark_tasks(model.eval(), Tokenizer.from_file(str(Path(config.data.tokenizer_dir) / "tokenizer.json")), device, config, log_dir, str(best_path))
    return None
