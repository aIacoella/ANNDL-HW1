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
labels[labels == "healthy"] = 0
labels[labels == "unhealthy"] = 1
labels = labels.astype(np.float32)

batch = data[0:50]

model = submission_model.model("SubmissionModel")

preds = model.predict(batch)

# Expected output:
correct_preictions = 0
for i in range(50):
    if preds[i] == labels[i]:
        correct_preictions += 1

print("Accuracy: " + str(correct_preictions / 50))

