# Iris Classification with a Neural Network

## Overview

This project implements a feed-forward neural network from scratch using
NumPy to classify the three Iris flower species.
It also includes
exploratory data analysis (EDA), model evaluation, a scikit-learn
baseline, and experiments with learning rates, activation functions, and
network architectures.

## Dataset

The project uses the Iris dataset from `sklearn.datasets.load_iris`.

- Samples: 150
- Input features: 4
- Classes: Setosa, Versicolor, Virginica
- Class distribution: 50 samples per class
- Missing values: None in the loaded dataset

The target values are encoded as `0`, `1`, and `2`, then converted to
one-hot vectors for training the custom neural network.

## Network architecture

The required custom network is:

`4 input features â 8 sigmoid neurons â 6 sigmoid neurons â 3 softmax outputs`

The implementation includes:

- Random weight initialization with fan-in scaling
- Sigmoid activation and its derivative
- Softmax output activation
- Forward propagation
- Categorical cross-entropy loss
- Backpropagation using the chain rule
- Gradient-descent parameter updates
- Mini-batch training with batch size 16
- Validation monitoring and early stopping

Default training configuration:

Parameter Value
------------------------- ---------------------------
Learning rate 0.01
Maximum epochs 1,000
Mini-batch size 16
Early-stopping patience 10 evaluation checkpoints
Evaluation interval Every 50 epochs
Random seed 42

## Project files

-----------------------------------------------------------------------
File Purpose
----------------------------------- -----------------------------------
`assignment.ipynb` Main notebook: data exploration,
training, evaluation, comparison,
and experiments

`data_preparation.py` Loads Iris, splits data, one-hot
encodes labels, scales features,
and creates EDA plots

`nn_pipeline.py` Implements the custom neural
network and training loop

`eval_analysis.py` Test evaluation, classification
report, confusion matrix, PCA
decision-boundary plot, and
scikit-learn comparison


`experiment_analysis.py`
Runs experiments on learning-rate,
activation-function, and
architecture, and
summarizes the best validation
results

`README.md`
Project overview and usage
instructions

`Work report.md`
Assignment work report and observed
results

-----------------------------------------------------------------------

Keep the notebook and Python modules in the same directory so the
imports work as written in the notebook.
The notebook currently imports
modules using the names `data_preparation`, `nn_pipeline`,
`eval_analysis`, and `experiment_analysis`; rename the uploaded files to
those exact names if they currently include version suffixes such as
`(4)`.


## Requirements

Install the required Python packages:

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## Run the project

1. Place the notebook and four Python modules in one directory.
2. Start Jupyter Notebook or JupyterLab from that directory.
3. Open `assignment.ipynb`.
4. Restart the kernel and run all cells from top to bottom.

The notebook performs EDA, creates a stratified train/validation/test
split, standardizes the features using statistics learned from the
training set, trains the custom network, evaluates it, compares it with
`MLPClassifier`, and runs the experiments.


## Data preparation

The split contains:

| Split Samples | Approximate | share |
|------------ |---------| ------------------- |
| Training | 104 | 69.3% |
| Validation | 23 | 15.3% |
| Test | 23 | 15.3% |
| Total | 150 | 100% |



Because 150 samples cannot be divided into exact integer counts of 70%,
15%, and 15%, these counts are the closest practical split produced by
the current two-stage stratified splitting process.


`StandardScaler` is fitted only on the training set, then applied to
validation and test sets to avoid leaking information from those sets
into preprocessing.


## Evaluation and experiments

The notebook includes:

- Test accuracy, classification report, and confusion matrix
- PCA-based visualization of the custom model's decision regions
- Comparison with scikit-learn's `MLPClassifier`
- Learning-rate experiments for `0.001`, `0.01`, and `0.1`
- Activation experiments using ReLU, tanh, and logistic/sigmoid
- Architecture experiments using `(6,)`, `(8, 6)`, and `(12, 8, 6)`
hidden-layer configurations

See `Work report.md` for the results recorded in the current notebook
run.


## Notes

- Experimental results depend on the data split, random
initialization, package versions, and training configuration.

- The experiment module reports test accuracy for each configuration.
Use validation performance to select a configuration and reserve the
test set for the final assessment rather than choosing a model based
on its test score.

- The custom model and the scikit-learn baseline use different
configurations, so their comparison is informative but not a
perfectly controlled architecture-only comparison.