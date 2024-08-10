import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from PIL import Image

# Define the path to your local images directory
import gzip
f = gzip.open('/content/drive/MyDrive/Colab Notebooks/train-labels-idx1-ubyte.gz','r')

image_size = 28
num_images = 5

import numpy as np
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)


# Function to load and preprocess images from a directory
def load_images_from_folder(folder, target_size=(28, 28)):
    loaded_images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert('L')  # Convert to grayscale
        img = img.resize(target_size)  # Resize to a common shape
        if img is not None:
            loaded_images.append(np.array(img))
    return np.array(loaded_images)


# Load and preprocess images
images = load_images_from_folder(images_directory)

# Assuming you have image labels or classes, if not, you need to label your data somehow
# For example, if you have subfolders for each class, you can use folder names as labels
# Replace this with your label generation logic
labels = np.array([0] * len(images))

# Check if labels are available
print(f"Number of labels: {len(labels)}")

# Normalize pixel values to be between 0 and 1
images = images / 255.0

# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Model definition
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Assuming you have 10 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)

# Make predictions
predictions = model.predict(test_images)

# Plot the first few test images, their predicted labels, and the true labels
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Predicted: {}, Actual: {}".format(np.argmax(predictions[i]), test_labels[i]))
plt.show()
