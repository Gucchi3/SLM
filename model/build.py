# @file build.py
# @brief Build a model from its name

##########################################################
'''
Add a model
1. Make a folder model/xxx/ and write the model class in model/xxx/xxx.py.
2. Add "name: class" to MODEL_REGISTRY.
3. Set config.model.name (or config.model.teacher_name) to that name.
A hybrid model (blocks of c, g, a, w) needs no class: add "name: HybridSpec(mixers=...)" with only the values that differ from the standard model.
'''
##########################################################

from dataclasses  import fields
from rich.console import Console
from rich.text    import Text
from torch        import nn
from types        import SimpleNamespace

import torch

from utils.exceptions import Error
from .glint2          import Glint2
from .hotaru          import Hotaru
from .hybrid          import HybridLm, HybridSpec, LoopSpec
from .supra_mini_v6   import SupraMiniV6
from .yokoyama        import Yokoyama

# The hybrid models are named by their blocks (c = gated short convolution, g = Gated DeltaNet, a = attention over all tokens, w = attention over the last 256 tokens; c3a = ccca),
# with a tag for each value that differs from the standard model (see HybridSpec). Their loop is set in config.model.loop.
MODEL_REGISTRY: dict[str, type[nn.Module] | HybridSpec] = {
    "glint2"       : Glint2,
    "hotaru"       : Hotaru,
    "supra_mini_v6": SupraMiniV6,
    "yokoyama"     : Yokoyama,
    "a4-kvh4"      : HybridSpec(mixers="aaaa", kv_heads=4),
    "c3a"          : HybridSpec(mixers="ccca"),
    "g3a"          : HybridSpec(mixers="ggga"),
    "c3w"          : HybridSpec(mixers="cccw"),
    "g3w"          : HybridSpec(mixers="gggw"),
    "c3a-relu2-rms": HybridSpec(mixers="ccca", mlp="relu2", norm="rms"),
    "c3w-relu2-rms": HybridSpec(mixers="cccw", mlp="relu2", norm="rms"),
    "g3a-relu2-rms": HybridSpec(mixers="ggga", mlp="relu2", norm="rms"),
    "g3w-relu2-rms": HybridSpec(mixers="gggw", mlp="relu2", norm="rms"),
}



def build_model(name: str, checkpoint: str | None, loop: SimpleNamespace | None = None) -> nn.Module:
    '''
    #* Make the model for the name (with the loop settings for a hybrid model), and load the weights when a checkpoint is given.
    #? Called from utils/trainer/pipeline.py for the student and the teacher, and from utils/validator/load.py.
    #! With a null checkpoint, the model starts from the default layer initialization.
    #! Weights whose names do not match (such as a head changed for another task) are skipped and listed. A layer with the same name and another shape stops with an error, so give a changed layer a new name.
    #! loop is config.model.loop (the keys of LoopSpec), or None for no loop. Only the hybrid models take it.
    '''
    
    # Find the model
    if name not in MODEL_REGISTRY:
        raise Error("Unknown model.", detail=f"model name: {name!r}\nAvailable: {', '.join(MODEL_REGISTRY)}", hint="Add the model to MODEL_REGISTRY in model/build.py.", code=6)
    entry = MODEL_REGISTRY[name]
    if loop is not None and not isinstance(entry, HybridSpec):
        raise Error("This model has no loop.", detail=f"model name: {name!r}", hint="Remove model.loop, or use a hybrid model such as c3a.", code=39)
    
    # Read and check the loop settings
    settings = LoopSpec()
    if loop is not None:
        keys    = [field.name for field in fields(LoopSpec)]
        unknown = sorted(set(vars(loop)) - set(keys))
        if unknown:
            raise Error("Unknown key in model.loop.", detail=f"keys: {', '.join(unknown)}\nAvailable: {', '.join(keys)}", hint="Fix the key names in model.loop.", code=40)
        settings = LoopSpec(**vars(loop))
    blocks = len(entry.mixers) if isinstance(entry, HybridSpec) else 0
    if not (0 <= settings.start <= settings.end <= blocks and settings.times >= 1):
        raise Error("Invalid model.loop.", detail=f"model.loop: {settings}\nblocks: {blocks}", hint="Use 0 <= start <= end <= blocks and times >= 1.", code=41)
    
    # Make the model
    model = HybridLm(entry, settings) if isinstance(entry, HybridSpec) else entry()
    if checkpoint is None:
        return model
    
    # Read the trained weights
    try:
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except OSError as error:
        raise Error("Could not load the checkpoint.", detail=f"{checkpoint}: {error}", hint="Check model.checkpoint and model.teacher_checkpoint, or set them to null.", code=8)
    
    # Load the weights whose names match
    try:
        result = model.load_state_dict(weights, strict=False)
    except RuntimeError as error:
        raise Error("The checkpoint does not fit the model.", detail=f"{checkpoint}: {error}", hint="A layer has the same name but another shape. Give the changed layer a new name, so its weights are skipped.", code=13)
    
    # List the weights that were not loaded
    console = Console()
    if result.missing_keys:
        console.print(Text(f"[Checkpoint] Not in {checkpoint}, so they keep the new initial values: {', '.join(result.missing_keys)}", style="yellow"))
    if result.unexpected_keys:
        console.print(Text(f"[Checkpoint] Not in the model {name}, so they are not used: {', '.join(result.unexpected_keys)}", style="yellow"))
    return model
