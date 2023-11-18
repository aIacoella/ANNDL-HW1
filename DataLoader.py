from sklearn.model_selection import train_test_split
from tensorflow import keras as tfk
import matplotlib.pyplot as plt
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

    print("Removing trolls, shreks and duplicates")
    initialDataN = data.shape[0]

    troll = data[338]
    shrek = data[58]

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(troll / 255)
    axes[1].imshow(shrek / 255)
    plt.tight_layout()
    plt.show()

    mask = []
    for i in range(data.shape[0]):
        if np.array_equal(data[i], troll) or np.array_equal(data[i], shrek):
            mask.append(False)
        else:
            mask.append(True)
    data = data[mask]
    y = y[mask]

    data, indexes = np.unique(data, axis=0, return_index=True)
    y = y[indexes]

    #y = tfk.utils.to_categorical(y, num_classes=2)

    print("Removed Images: " + str(initialDataN - data.shape[0]))

    if(test_size==0 and val_size==0):
        return (data, y), (None, None), (None, None)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        data, y, random_state=seed, test_size=test_size, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, random_state=seed, test_size=val_size, stratify=y_train_val)

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)
