import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    data = load_iris()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df["Target"] = data.target
    return df


def one_hot_encode(y, num_classes: int = 3) -> np.ndarray:
    """Converts 1D integer class labels into a 2D one-hot encoded NumPy array."""
    y_arr = np.asarray(y, dtype=int).ravel()
    return np.eye(num_classes)[y_arr]


def split_data(
    data: pd.DataFrame,
    target_col: str = "Target",
    test_size: float = 0.15,
    val_size: float = 0.15,
    num_classes: int = 3,
    encode_one_hot: bool = True,
    random_state: int = 42,
):
    """Splits DataFrame into Train, Validation, and Test sets and encodes labels to one-hot vectors."""
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Step 1: Split into 85% train+val and 15% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Step 2: Split 85% into train (70% of total) and val (15% of total)
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_train,
    )

    # Convert labels to one-hot encoded vectors
    if encode_one_hot:
        y_train = one_hot_encode(y_train, num_classes=num_classes)
        y_val = one_hot_encode(y_val, num_classes=num_classes)
        y_test = one_hot_encode(y_test, num_classes=num_classes)

    total = len(data)
    print("=" * 10, "Data Split Summary", "=" * 10)
    print(f"Train:      {X_train.shape[0]} samples ({X_train.shape[0] / total:.1%}) | y shape: {y_train.shape}")
    print(f"Validation: {X_val.shape[0]} samples ({X_val.shape[0] / total:.1%}) | y shape: {y_val.shape}")
    print(f"Test:       {X_test.shape[0]} samples ({X_test.shape[0] / total:.1%}) | y shape: {y_test.shape}")

    # Standard convention: Train -> Val -> Test
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    """Fits StandardScaler on X_train only and transforms train, val, and test sets."""
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def feature_dist(data: pd.DataFrame):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    cols = data.columns

    for i in range(2):
        for j in range(2):
            col_idx = i * 2 + j
            col_name = cols[col_idx]
            sns.histplot(
                data=data,
                x=col_name,
                ax=axes[i, j],
                kde=True,
                color="steelblue"
            )
            axes[i, j].set_title(f"Distribution of {col_name}")

    plt.tight_layout()
    plt.show()


def feature_pair(data: pd.DataFrame, hue=None):
    sns.pairplot(
        data=data,
        hue=hue,
        palette="Set2"
    )
    plt.show()