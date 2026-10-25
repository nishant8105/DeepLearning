import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def run_experiments(X_train, y_train, X_val, y_val, X_test, y_test):

    # Convert one-hot labels into class numbers
    if y_train.ndim == 2:
        y_train = np.argmax(y_train, axis=1)
    if y_val.ndim == 2:
        y_val = np.argmax(y_val, axis=1)
    if y_test.ndim == 2:
        y_test = np.argmax(y_test, axis=1)

    results = []

    # Different learning rates
    for lr in [0.001, 0.01, 0.1]:
        model = MLPClassifier(
            hidden_layer_sizes=(8, 6),
            activation="logistic",
            learning_rate_init=lr,
            solver="sgd",
            max_iter=2000,
            random_state=42
        )

        model.fit(X_train, y_train)

        results.append({
            "Experiment": "Learning Rate",
            "Setting": lr,
            "Train Accuracy": accuracy_score(y_train, model.predict(X_train)),
            "Validation Accuracy": accuracy_score(y_val, model.predict(X_val)),
            "Test Accuracy": accuracy_score(y_test, model.predict(X_test))
        })

    # Different activation functions
    for act in ["relu", "tanh", "logistic"]:
        model = MLPClassifier(
            hidden_layer_sizes=(8, 6),
            activation=act,
            learning_rate_init=0.01,
            solver="sgd",
            max_iter=2000,
            random_state=42
        )

        model.fit(X_train, y_train)

        results.append({
            "Experiment": "Activation",
            "Setting": act,
            "Train Accuracy": accuracy_score(y_train, model.predict(X_train)),
            "Validation Accuracy": accuracy_score(y_val, model.predict(X_val)),
            "Test Accuracy": accuracy_score(y_test, model.predict(X_test))
        })

    # Different network architectures
    architectures = [
        (6,),
        (8, 6),
        (12, 8, 6)
    ]

    for layers in architectures:
        model = MLPClassifier(
            hidden_layer_sizes=layers,
            activation="logistic",
            learning_rate_init=0.01,
            solver="sgd",
            max_iter=2000,
            random_state=42
        )

        model.fit(X_train, y_train)

        results.append({
            "Experiment": "Architecture",
            "Setting": str(layers),
            "Train Accuracy": accuracy_score(y_train, model.predict(X_train)),
            "Validation Accuracy": accuracy_score(y_val, model.predict(X_val)),
            "Test Accuracy": accuracy_score(y_test, model.predict(X_test))
        })

    # Create comparison table
    results = pd.DataFrame(results)

    return results


def show_best_results(results):

    best_results = results.loc[
        results.groupby("Experiment")["Validation Accuracy"].idxmax()
    ]

    print("Best results for each experiment:")
    print(best_results.to_string(index=False))

    return best_results
