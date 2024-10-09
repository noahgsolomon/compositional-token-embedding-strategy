# Fine-Tuning GPT-2 with New Tokens

This project explores two methods to integrate new tokens into a pre-trained GPT-2 language model:

1. **Tokens Anew (`tokens_anew.py`)**: Adds new tokens with randomly initialized embeddings.
2. **Tokens Mapped (`tokens_mapped.py`)**: Maps new tokens as combinations of existing embeddings using a custom embedding layer.

- Extract sentences containing "love", "hate", or "fear".
- Replace these words with `[LOVEEE]`, `[HATEEE]`, or `[FEARR]`.
- Use a subset of 1000 sentences for demonstration.

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Datasets
- Matplotlib

## Usage

Run `tokens_anew.py`
Run `tokens_mapped.py`

Run `comparison.py`

and then check the `loss_comparison.png` for the loss comparison of these two fine tuning methods
