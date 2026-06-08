
# Progamming Practical 1: Sampling from GPT2

This Programming Practical focuses on sampling from a trained GPT-2 model. After completing the missing part of `model.py`, we will use the `sample.py` script to generate text samples based on a given prompt. 

This code is a skinny version of Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master?tab=readme-ov-files), focused on sampling and GPT2 only. Check out the full repo for more features and training code. 

## Complete the Missing Parts

In this PP, you will mainly implement the missing parts of several building blocks of `model.py`. the goal is to get a working GPT-2 implementation that can sample text. First, take a look at the code to understand the overall structure and flow. 

You will have to use config parameters, whose name can be found in the `@dataclass`GPTConfig, on l. 136:

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )
```
Make sure to use layers defined in the `__init__` method if they are declared. Let's go!

### `CausalSelfAttention`

In `model.py:30`, `CausalSelfAttention` implements multi-head causal self-attention (projection to q/k/v, masked attention, then output projection).

```python
def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    # key, query, value projections for all heads, but in a batch
    self.c_attn = # TODO. /!\ note that each k, q, v vector will be of size (n_embd // n_head)
    # output projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
```

- Complete `self.c_attn`.
- Make each key, query, and value vector use the per-head embedding size.

```python
def forward(self, x):
    B, T, C = (
        x.size()
    )  # batch size, sequence length, embedding dimensionality (n_embd)

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = # TODO. output shape: (B, nh, T, hs)
    q = # TODO. output shape:  (B, nh, T, hs)
    v = # TODO. output shape: (B, nh, T, hs)

    # [...]
    else:
        # manual implementation of attention
        att = # TODO matmul and scaling
        att = # TODO causal mask (using att.masked_fill)
        att = # TODO softmax and dropout
        y = att @ v
    y = (
        y.transpose(1, 2).contiguous().view(B, T, C)
    )  # re-assemble all head outputs side by side

    # output projection
    y = self.resid_dropout(
        # TODO
        )
    return y
```

- Reshape `k`, `q`, and `v` to the required output shape `(B, nh, T, hs)`.
- In the non-flash branch, compute attention scores with matrix multiplication and scaling.
- Apply a causal mask using `att.masked_fill`.
- Apply softmax, then attention dropout.
- Fill the missing lines inside `self.resid_dropout(...)` for the output projection path.

### `MLP`

In `model.py:98`, `MLP` is the feed-forward sub-layer used inside each transformer block.

```python
def __init__(self, config):
    super().__init__()
    self.c_fc = # TODO the depth of the MLP will be 4 * config.n_embd
    self.gelu = nn.GELU()
    self.c_proj = # TODO
    self.dropout = nn.Dropout(config.dropout)
```

- Complete `self.c_fc` and follow the explicit TODO: `"the depth of the MLP will be 4 * config.n_embd"`.
- Set `self.c_fc` so the hidden depth is `4 * config.n_embd`.
- Complete `self.c_proj`.

```python
def forward(self, x):
    # TODO
    return x
```

- Complete the missing lines in the forward pass.

### `Block`

In `model.py:112`, `Block` combines normalization, attention, and MLP with residual connections.

```python
def forward(self, x):
    # TODO layer_norm -> attention with residual connection -> layer_norm -> MLP with residual connection
    return x
```

- Implement this order in the missing lines: layer norm, attention with residual, layer norm, then MLP with residual.

### `GPT`

In `model.py:141`, `GPT` defines the full language model and, in `forward`, applies embeddings, stacked blocks, and output logits.

```python
def forward(self, idx):
    # [...]

    # forward the GPT model itself
    tok_emb = self.transformer.wte(idx)
    pos_emb = self.transformer.wpe(pos)
    x = # TODO: Dropout + positional encoding
    # TODO apply blocks sequentially
    x = self.transformer.ln_f(x)

    logits = self.lm_head(
        # TODO keep the logits of the final position
    )

    return logits
```

- Build `x` from token embeddings and positional embeddings, then apply dropout.
- Apply each transformer block sequentially.
- Keep only the logits from the final time position.


## Sample Text

Once you have completed the implementation, use `sample.py` to test it by generating text samples from from pre-trained GPT-2.

!!! Note
    It is likely that your implementation will fail at first, you will have to make back and forth between sampling and debugging the code in previous section.

### Basic Usage

```bash
# Sample from GPT-2
uv run python sample.py --start="Hello, my name is"
```

### CLI Options

All available command-line options for `sample.py`:

- `--start`: The prompt text to start generation (default: `"\n"`)
    - Can be any string: `--start="Once upon a time"`
    - Can load from a file: `--start="FILE:prompt.txt"`
    - Special tokens like `--start="<|endoftext|>"`

- `--num_samples`: Number of samples to generate (default: `1`)

- `--max_new_tokens`: Number of tokens to generate per sample (default: `100`)

- `--temperature`: Sampling temperature (default: `0.8`)
    - `1.0`: No change (standard sampling)
    - `< 1.0`: Less random (more deterministic)
    - `> 1.0`: More random (more creative)

- `--top_k`: Keep only top k most likely tokens (default: `200`)
    - Higher values: More diverse outputs
    - Lower values: More focused outputs

- `--seed`: Random seed for reproducibility (default: `1337`)

- `--device`: Device to run on (default: `"cpu"`)
    - Examples: `"cpu"`, `"cuda"`, `"cuda:0"`, `"cuda:1"`


## Love Letter

Try to generate a decent love letter using GPT2. Then, send it to [pnovello@insa-toulouse.fr](mailto:pnovello@insa-toulouse.fr). The more romantic one, the funniest one and the clumsiest one will be displayed in the website. Work hard for posterity!

## Bonus: Implement Layer Norm by Yourself

In `model.py`, l. 27, layer norm is implemented by calling `torch.nn.LayerNorm`. 

```python
def forward(self, input):
    return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

Try to implement it by yourself, without using PyTorch's built-in implementation. 
