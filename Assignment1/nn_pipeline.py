
import numpy as np


# 1. Initialize parameters
def initialize_parameters(n_x, n_h1, n_h2, n_y=3, seed=42):
    rng = np.random.default_rng(seed)

    parameters = {
        "W1": rng.standard_normal((n_x, n_h1)) * np.sqrt(1 / n_x),
        "b1": np.zeros((1, n_h1)),
        "W2": rng.standard_normal((n_h1, n_h2)) * np.sqrt(1 / n_h1),
        "b2": np.zeros((1, n_h2)),
        "W3": rng.standard_normal((n_h2, n_y)) * np.sqrt(1 / n_h2),
        "b3": np.zeros((1, n_y)),
    }

    return parameters


# 2. Sigmoid and derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(a):
    return a * (1 - a)


# 3. Softmax
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# 4. Forward propagation
def forward_propagation(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)

    cache = (X, A1, A2, A3)

    return A3, cache


# 5. Compute loss
def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return loss


# 6. Backward propagation
def backward_propagation(parameters, cache, y_true):
    X, A1, A2, A3 = cache
    m = X.shape[0]

    dZ3 = (A3 - y_true) / m
    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dZ2 = (dZ3 @ parameters["W3"].T) * sigmoid_derivative(A2)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = (dZ2 @ parameters["W2"].T) * sigmoid_derivative(A1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    gradients = {
        "dW1": dW1, "db1": db1,
        "dW2": dW2, "db2": db2,
        "dW3": dW3, "db3": db3
    }

    return gradients


# 7. Update parameters
def update_parameters(parameters, gradients, learning_rate):
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        parameters[key] -= learning_rate * gradients["d" + key]

    return parameters


# 8. Predict and calculate accuracy
def predict(X, parameters):
    y_pred, _ = forward_propagation(X, parameters)
    return np.argmax(y_pred, axis=1)


def accuracy_score(y_true, y_pred):
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)

    return np.mean(y_true == y_pred)


# 9. Train model
def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    n_h1=8,
    n_h2=6,
    learning_rate=0.01,
    epochs=1000,
    batch_size=16,
    patience=10,
    seed=42,
):
    rng = np.random.default_rng(seed)

    parameters = initialize_parameters(
        X_train.shape[1],
        n_h1,
        n_h2,
        y_train.shape[1],
        seed=seed,
    )

    best_parameters = {
        k: v.copy() for k, v in parameters.items()
    }
    best_val_accuracy = -np.inf
    wait = 0

    history = {
        "epochs": [],
        "loss": [],
        "val_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        # Shuffle indices without changing the original arrays
        indices = rng.permutation(len(X_train))

        for start in range(0, len(X_train), batch_size):
            batch_indices = indices[start:start + batch_size]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            y_pred, cache = forward_propagation(
                X_batch, parameters
            )

            gradients = backward_propagation(
                parameters, cache, y_batch
            )

            parameters = update_parameters(
                parameters, gradients, learning_rate
            )

        # Evaluate every 50 epochs
        if epoch % 50 == 0:
            train_pred, _ = forward_propagation(
                X_train, parameters
            )
            val_pred, _ = forward_propagation(
                X_val, parameters
            )

            loss = compute_loss(y_train, train_pred)
            val_accuracy = accuracy_score(
                y_val, np.argmax(val_pred, axis=1)
            )

            history["epochs"].append(epoch)
            history["loss"].append(loss)
            history["val_accuracy"].append(val_accuracy)

            print(
                f"Epoch {epoch:4d} | "
                f"Loss: {loss:.4f} | "
                f"Validation Accuracy: {val_accuracy:.4f}"
            )

            # Early stopping based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_parameters = {
                    k: v.copy()
                    for k, v in parameters.items()
                }
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                print("Early stopping: validation accuracy plateaued.")
                break

    # Restore parameters from the best validation checkpoint
    return best_parameters, history