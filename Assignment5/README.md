# MNIST Autoencoders

This assignment compares several autoencoder approaches for MNIST digit
images. The notebook loads the dataset and demonstrates a vanilla
fully-connected autoencoder, a convolutional autoencoder, a denoising
autoencoder, and a variational autoencoder (VAE). It visualizes
reconstructions, tests denoising at different noise levels, and generates
digits from sampled latent vectors.

## Files

- `Assignment5.ipynb` — the complete assignment workflow.
- `data/MNIST/` — local MNIST data, if already downloaded. Otherwise,
  `torchvision` downloads it when the notebook runs.

## Requirements

```bash
pip install torch torchvision matplotlib jupyter
```

## Run

Run Jupyter from this folder, open `Assignment5.ipynb`, and execute the cells
in order. The notebook automatically selects CPU or GPU based on availability.
Downloaded dataset files are local data and may be excluded from Git.
