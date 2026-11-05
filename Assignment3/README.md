# Shakespeare Character-Level Text Generation

This assignment trains a character-level LSTM in PyTorch to predict and
generate Shakespeare-style text. The notebook prepares the text corpus,
creates character-to-index mappings and fixed-length sequences, trains the
model, plots training and validation loss, and generates samples with
temperature, top-k, and top-p sampling.

## Files

- `Assignment3.ipynb` — the complete data, training, and generation workflow.
- `shakespeare.txt` — the text corpus used by the notebook. If it is missing,
  the notebook downloads it.
- `model_epoch_5.pth`, `model_epoch_10.pth`, `model_epoch_15.pth`,
  `model_epoch_20.pth`, `model_epoch_25.pth`, and `model_epoch_30.pth` — saved
  model checkpoints.

## Requirements

```bash
pip install torch numpy matplotlib jupyter
```

## Run

Run Jupyter from this folder, open `Assignment3.ipynb`, and execute its cells
in order. Training uses the available CPU or GPU. The notebook trains for 30
epochs by default and includes an interactive cell for entering a prompt and
generation settings.
