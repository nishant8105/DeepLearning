# Deep Learning Assignments

This repository contains six independent deep-learning assignments. Each
assignment has its own notebook and README with its topic, data, and run
instructions. Open the notebook in the corresponding folder and run its cells
from top to bottom.

## Assignments

| Folder | Topic | Main notebook |
|---|---|---|
| [Assignment1](./Assignment1/README.md) | Iris classification with a neural network implemented in NumPy | [assignment.ipynb](./Assignment1/assignment.ipynb) |
| [Assignment2](./Assignment2/README.md) | CIFAR-10 image classification with PyTorch | [assignment.ipynb](./Assignment2/assignment.ipynb) |
| [Assignment3](./Assignment3/README.md) | Character-level Shakespeare text generation with an LSTM | [Assignment3.ipynb](./Assignment3/Assignment3.ipynb) |
| [Assignment4](./Assignment4/README.md) | Customer segmentation with a Self-Organizing Map (SOM) | [assignment4.ipynb](./Assignment4/assignment4.ipynb) |
| [Assignment5](./Assignment5/README.md) | MNIST autoencoders | [Assignment5.ipynb](./Assignment5/Assignment5.ipynb) |
| [Assignment6](./Assignment6/README.md) | Fashion-MNIST autoencoders and classifiers | [Assignment6.ipynb](./Assignment6/Assignment6.ipynb) |

## Getting started

There is no shared environment or dependency manifest for the repository.
Install the packages listed in the README for the assignment you want to run.
PyTorch assignments can use a CUDA-capable GPU when available; the notebooks
also support CPU execution, though training may take longer.

For each assignment:

1. Change to its folder.
2. Install its listed dependencies in your Python environment.
3. Start Jupyter and open the notebook named in the table above.
4. Run the notebook from top to bottom.

Some notebooks download datasets when needed. The `.gitignore` excludes
directories named `data`, so downloaded or locally stored datasets may not be
included in a clone of this repository. Assignment 4's source CSV is stored
directly in its folder.

## Repository layout

- `Assignment1/` contains the Iris data-preparation, neural-network, evaluation,
  and experiment modules, the notebook, and a work report.
- `Assignment2/` contains the CIFAR-10 notebook and dataset files.
- `Assignment3/` contains the text-generation notebook, Shakespeare text, and
  saved model checkpoints.
- `Assignment4/` contains the customer-segmentation notebook, source data, and
  exported segment and SOM artifacts.
- `Assignment5/` contains the MNIST autoencoder notebook and local MNIST data.
- `Assignment6/` contains the Fashion-MNIST autoencoder/classifier notebook.
