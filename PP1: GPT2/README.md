
# Progamming Practical 1: Sampling from GPT2

This Programming Practical focuses on sampling from a trained GPT-2 model. After completing the missing part of `model.py`, we will use the `sample.py` script to generate text samples based on a given prompt. 

This code is a skinny version of Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master?tab=readme-ov-files), focused on sampling and GPT2 only. Check out the full repo for more features and training code.


## Sampling from a Trained Model

Use `sample.py` to generate text samples from a trained model or from pre-trained GPT-2 variants.

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
