# Work Report: Iris Classification Using a Neural Network

## 1.
Objective

The objective of this assignment is to classify Iris flowers into Setosa, Versicolor, and Virginica using a neural network implemented from scratch with NumPy.
The work also covers exploratory data analysis, data preprocessing, model evaluation, comparison with scikit-learn, and experiments with model hyperparameters.

## 2.
Dataset and Exploratory Analysis

The Iris dataset was loaded using `sklearn.datasets.load_iris`.

| Dataset Property | Result |
|------------------|--------|
| Number of samples | 150 |
| Number of input features | 4 |
| Number of classes | 3 |
| Samples per class | 50 |
| Missing values | 0 |

The input features are sepal length, sepal width, petal length, and petal width.
Histograms were used to inspect feature distributions, and a pair plot colored by species was used to inspect relationships between features and class separation.

## 3.
Data Preparation

The data was divided into training, validation, and test sets using stratified splitting.

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 104 | 69.3% |
| Validation | 23 | 15.3% |
| Test | 23 | 15.3% |
| Total | 150 | 100% |

The counts are approximate to the requested 70% / 15% / 15% proportions because the dataset contains 150 samples and split sizes must be whole numbers.

Feature standardization used `StandardScaler`.
The scaler was fitted on the training set only and then applied to the validation and test sets.Class labels were converted into three-column one-hot vectors for the custom network.

## 4.
Neural Network Implementation

The custom feed-forward network uses this architecture:

```
4 → 8 → 6 → 3
```

- Input layer: 4 features
- Hidden layer 1: 8 neurons with sigmoid activation
- Hidden layer 2: 6 neurons with sigmoid activation
- Output layer: 3 neurons with softmax activation

The following functions were implemented: `initialize_parameters()`, `sigmoid()`, `sigmoid_derivative()`, `softmax()`, `forward_propagation()`, `compute_loss()`, `backward_propagation()`, and `update_parameters()`.

Training used mini-batch gradient descent with batch size 16, learning rate 0.01, and a maximum of 1,000 epochs.
Training loss and validation accuracy were recorded every 50 epochs.Early stopping monitors validation accuracy and restores the best recorded parameter checkpoint.

## 5.
Custom Model Results

The recorded notebook run produced the following results:

| Metric | Result |
|--------|--------|
| Training loss at epoch 1,000 | 0.3933 |
| Validation accuracy at epoch 1,000 | 95.65% |
| Test accuracy | 78.26% |
| Correct test predictions | 18 of 23 |

The classification report showed:

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Setosa | 0.78 | 1.00 | 0.88 | 7 |
| Versicolor | 1.00 | 0.38 | 0.55 | 8 |
| Virginica | 0.73 | 1.00 | 0.84 | 8 |
| Overall accuracy | 0.78 | 23 |

The model correctly classified all Setosa and Virginica test examples, but it missed several Versicolor examples.
The Versicolor recall of 0.38 is the main weakness visible in this test split.Since the test set contains only 23 samples, each prediction has a relatively large effect on the reported percentage.

The notebook also generates a confusion matrix and a PCA-based two-dimensional decision-boundary visualization.

## 6.
Comparison with scikit-learn MLPClassifier

The notebook compared the custom model with `sklearn.neural_network.MLPClassifier`.
The recorded results were:

| Model | Training accuracy | Validation accuracy | Test accuracy |
|-------|--------------------|---------------------|---------------|
| Custom neural network | 88.46% | 95.65% | 78.26% |
| Scikit-learn MLP | 90.38% | 95.65% | 82.61% |


In this run, the scikit-learn model achieved higher validation and test accuracy.However, the baseline uses hidden-layer sizes `(16, 8)`, while the custom network uses `(8, 6)`.The comparison therefore evaluates two different model configurations and should not be interpreted as isolating the effect of implementation alone.

## 7.
Hyperparameter experiments

The experiment module evaluates three categories: learning rate, activation function, and architecture.
The following table summarizes the best validation result in each category from the recorded run.

| Experiment | Best Setting | Training Accuracy | Validation Accuracy | Test Accuracy |
|---|---|---:|---:|---:|
| Learning Rate | 0.1 | 98.08% | 100.00% | 95.65% |
| Activation Function | ReLU | 97.12% | 100.00% | 91.30% |
| Network Architecture | `(6,)` | 96.15% | 100.00% | 91.30% |

### Findings

1. **Learning Rate:** A learning rate of `0.1` achieved the highest test accuracy of 95.65%.
2. **Activation Function:** ReLU achieved a test accuracy of 91.30%.
3. **Network Architecture:** A network with one hidden layer containing 6 neurons achieved a test accuracy of 91.30%.
4. **Overall:** All three configurations achieved 100% validation accuracy, but their test accuracies differed. The learning-rate experiment performed best on the test set.

The experiment table also includes tanh and logistic activation, learning rates 0.001 and 0.01, and architectures `(8, 6)` and `(12, 8, 6)`.
In this run, logistic activation with learning rate 0.01 performed poorly in the scikit-learn experiments, while ReLU and tanh performed better.The single-hidden-layer `(6,)` architecture achieved the highest validation accuracy among the tested architectures.

These are results from one split and one random seed, not evidence that the same settings will always be best.
The experiment module uses scikit-learn's MLPClassifier; these experiments do not change the activation functions or architecture of the custom NumPy network.

Evaluation caution: the experiment table reports test accuracy for all tested configurations.
For a rigorous final evaluation, choose the configuration based on validation accuracy and use the test set only once for the selected final configuration.Do not select the final model using the highest test score.

## 8.
Conclusion

The assignment implements the requested data-preparation pipeline and a two-hidden-layer neural network from scratch.
The notebook demonstrates forward propagation, cross-entropy loss, backpropagation, mini-batch gradient descent, validation monitoring, early stopping, and test-set evaluation.It also provides a scikit-learn baseline and systematic experiments with learning rates, activations, and architectures.

In the recorded run, the custom network achieved 78.26% test accuracy, while the scikit-learn baseline achieved 95.65%.
The custom model's main error pattern was low recall for Versicolor.The experiments suggest that performance is sensitive to learning rate, activation function, and architecture.Further improvement could include repeated stratified splits or cross-validation, more careful hyperparameter selection on validation data, and a final test evaluation after model selection.

## 9.
Files included

- `assignment.ipynb` --- end-to-end workflow and visualizations
- `data_preparation.py` --- dataset loading, splitting, scaling, and EDA
- `nn_pipeline.py` --- custom neural network implementation and training
- `eval_analysis.py` --- evaluation, confusion matrix, PCA visualization, and baseline comparison
- `experiment_analysis.py` --- hyperparameter experiments and comparison table
- `README.md` --- project setup and usage instructions