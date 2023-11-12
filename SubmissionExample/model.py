import os
import tensorflow as tf
import numpy as np


class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(
            os.path.join(path, 'SundayModel'))

    def predict(self, X):
        # Note: this is just an example.
        # Here the model.predict is called, followed by the argmax
        out = self.model.predict(X)
        out = np.where(out > 0, 1, 0).flatten()

        return out