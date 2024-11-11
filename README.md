# Token Recycling

Implementation of "Turning Trash into Treasure: Accelerating Inference of Large Language Models with Token Recycling" [arxiv](https://arxiv.org/abs/2408.08696).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.model.token_recycling import TokenRecycling

model = TokenRecycling(base_model="vicuna-7b")
output = model.generate("Your prompt here")
```
