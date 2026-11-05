# Fashion-MNIST: Autoencoder and Classifier

This capstone notebook compares unsupervised feature learning and image
classification on Fashion-MNIST. It trains vanilla, convolutional, denoising,
variational, and contractive autoencoders; visualizes reconstructions and
latent features; and compares classification approaches using raw images or
learned encoder representations, including frozen-encoder, fine-tuning, and
joint reconstruction/classification approaches.

The notebook starts in quick mode with a convolutional autoencoder. Set
`RUN_FULL_AE_GRID = True` in the notebook to run all five autoencoder types
across the configured latent sizes; this requires substantially more time.

## Files

- `Assignment6.ipynb` — the complete assignment workflow.
- `data/` — Fashion-MNIST data downloaded by `torchvision` as needed. Local
  dataset files may be excluded from Git.

## Requirements

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn umap-learn jupyter
```

## Run

Run Jupyter from this folder, open `Assignment6.ipynb`, and execute the cells
in order. A GPU is recommended for full training runs; the notebook also
supports CPU execution.
