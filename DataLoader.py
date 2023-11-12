from sklearn.model_selection import train_test_split
from tensorflow import keras as tfk
import numpy as np
seed = 42


def load_data(seed=42, path="./public_data.npz", test_size=0.2, val_size=0.2):
    np.random.seed(seed)
    data_file = np.load(path, allow_pickle=True)
    data = data_file["data"]
    y = data_file["labels"]
    y[y == "healthy"] = 0
    y[y == "unhealthy"] = 1
    y = y.astype(np.float32)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        data, y, random_state=seed, test_size=test_size, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, random_state=seed, test_size=val_size, stratify=y_train_val)

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)
