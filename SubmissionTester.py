import SubmissionModel.model as submission_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras import layers as tfkl
from tensorflow import keras as tfk
import numpy as np

seed = 42
np.random.seed(seed)

data_file = np.load('./public_data.npz', allow_pickle=True)

data = data_file["data"]
labels = data_file["labels"]

batch = data[0:10]

model = submission_model.model("SubmissionModel")

preds = model.predict(batch)
print(preds)

# Expected output:
print("Expected output:" + labels[0:10])
