from .hybrid_lm import HybridLm
from .types     import HybridSpec, LoopSpec

'''
hybrid/
├── __init__.py
├── types.py
│   ├── HybridSpec
│   └── LoopSpec
├── gated_delta.py
│   └── gated_delta_chunk()
├── hybrid_block.py
│   └── HybridBlock
│       └── forward()
└── hybrid_lm.py
    └── HybridLm
        └── forward()
'''
