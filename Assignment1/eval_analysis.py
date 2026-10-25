import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)
from sklearn.neural_network import MLPClassifier

def evaluate_model(X_test, y_test, parameters, predict_fn):
    y_pred = predict_fn(X_test, parameters)

    if y_test.ndim == 2:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    accuracy = np.mean(y_true == y_pred)

    print("Test Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Setosa", "Versicolor", "Virginica"],
        zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Setosa", "Versicolor", "Virginica"]
    )
    display.plot()
    plt.title("Confusion Matrix")
    plt.show()

    return accuracy, cm


def plot_pca_decision_boundaries(X_train, y_train, parameters, predict_fn):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_train)

    if y_train.ndim == 2:
        y = np.argmax(y_train, axis=1)
    else:
        y = y_train

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.05),
        np.arange(y_min, y_max, 0.05)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid)

    predictions = predict_fn(grid_original, parameters)
    predictions = predictions.reshape(xx.shape)

    plt.figure(figsize=(8, 5))
    plt.contourf(xx, yy, predictions, alpha=0.3)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolors="black")

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Decision Boundaries")
    plt.show()


def compare_with_sklearn_mlp(
    X_train, y_train, X_val, y_val,
    X_test, y_test, parameters, predict_fn
):
    def get_labels(y):
        if y.ndim == 2:
            return np.argmax(y, axis=1)
        return y

    y_train = get_labels(y_train)
    y_val = get_labels(y_val)
    y_test = get_labels(y_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="logistic",
        max_iter=1000,
        random_state=42
    )
    mlp.fit(X_train, y_train)

    custom_train = predict_fn(X_train, parameters)
    custom_val = predict_fn(X_val, parameters)
    custom_test = predict_fn(X_test, parameters)

    sklearn_train = mlp.predict(X_train)
    sklearn_val = mlp.predict(X_val)
    sklearn_test = mlp.predict(X_test)

    results = {
        "Model": ["Custom Neural Network", "Sklearn MLP"],
        "Training Accuracy": [
            accuracy_score(y_train, custom_train),
            accuracy_score(y_train, sklearn_train)
        ],
        "Validation Accuracy": [
            accuracy_score(y_val, custom_val),
            accuracy_score(y_val, sklearn_val)
        ],
        "Test Accuracy": [
            accuracy_score(y_test, custom_test),
            accuracy_score(y_test, sklearn_test)
        ]
    }

    results = pd.DataFrame(results)
    print(results)

    return results, mlp