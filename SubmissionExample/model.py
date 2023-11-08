import os
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras import layers as tfkl


class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(
            os.path.join(path, 'TransferLearningModel'))
        self.resizing = tfk.Sequential([
            tfkl.experimental.preprocessing.Resizing(224, 224)
        ])

    def predict(self, X):

        # Note: this is just an example.
        # Here the model.predict is called, followed by the argmax
        out = self.model.predict(preprocess_input(self.resizing(X)))
        out = tf.argmax(out, axis=-1)  # Shape [BS]

        return out
