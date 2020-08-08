# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

tf.keras.backend.clear_session()

# initialize parameters
EPOCHS = 20
BATCH_SIZE = 32
INIT_LR = 1e-4
HEIGHT, WIDTH, RGB = 224, 224, 3

# preprocessing
DIRECTORY = "./Dataset/"
CATEGORIES = ["with_mask", "without_mask"]
print("[INFO] loading images...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(HEIGHT, WIDTH))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

# split data set
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)
# data augmentation
train = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# model building
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(HEIGHT, WIDTH, RGB)))
baseModel.trainable = False

headModel = baseModel.output
model = Sequential([
    baseModel,
    AveragePooling2D(pool_size=(7, 7)),
    Flatten(name="flatten"),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(2, activation="softmax")
])

model.summary()

# compilation
print("[INFO] compiling model...")
optimizer = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# training
print("[INFO] training head...")
H = model.fit_generator(train.flow(trainX, trainY, batch_size=BATCH_SIZE),
                        steps_per_epoch=len(trainX) // BATCH_SIZE,
                        validation_data=(testX, testY),
                        validation_steps=len(testX) // BATCH_SIZE,
                        epochs=EPOCHS)
# save the model
print("[INFO] saving mask detector model...")
model.save("MaskDetector.model", save_format="h5")

N = EPOCHS

# visualisation
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training loss and Accuaracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
