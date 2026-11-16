from .build_hybrid import build_hybrid_model
from .hybrid_lm    import HybridLm
from .types        import HybridSpec

'''
hybrid/
├── __init__.py
├── types.py
│   ├── EmbedSpec
│   ├── PosSpec
│   ├── MlaSpec
│   ├── KvSpec
│   ├── AttnSpec
│   ├── ConvSpec
│   ├── GdnSpec
│   ├── FfnSpec
│   ├── NormSpec
│   ├── LoopSpec
│   └── HybridSpec
├── build_hybrid.py
│   ├── build_hybrid_model()
│   └── _build_hybrid_spec()
├── gated_delta.py
│   └── gated_delta_chunk()
├── int_norm.py
│   └── IntNorm
│       └── forward()
├── hybrid_block.py
│   └── HybridBlock
│       └── forward()
└── hybrid_lm.py
    └── HybridLm
        └── forward()
'''
