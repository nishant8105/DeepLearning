# CIFAR-10 Image Classification

This assignment uses PyTorch to train and evaluate a convolutional neural
network on CIFAR-10. The notebook covers loading and inspecting the images,
preprocessing and augmentation, building data loaders and a baseline CNN,
training, and evaluating results with accuracy, a confusion matrix, and
per-class metrics.

## Files

- `assignment.ipynb` — the complete assignment workflow.
- `data/` — local CIFAR-10 files. The notebook can download the dataset using
  `torchvision` if it is not already available.

The repository ignores directories named `data`, so dataset files may be
absent from a fresh clone.

## Requirements

Install the dependencies in the Python environment used by Jupyter:

```bash
pip install torch torchvision matplotlib numpy scikit-learn jupyter
```

A CUDA-capable GPU is recommended for longer training runs, but the notebook
can run on CPU. Start with a small number of epochs to check the environment
before running a longer experiment.

## Run

From this folder, start Jupyter, open `assignment.ipynb`, and run the cells in
order. Dataset downloads and training time depend on your network and hardware.
