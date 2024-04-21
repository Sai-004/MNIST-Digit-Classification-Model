import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten the images into 1D arrays
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# Convert labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Print dataset information
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)
print("Number of unique classes:", len(set(y_train)))

# Define the larger model with more dense layers
model = Sequential([
    Dense(100, input_shape=(784,), activation='relu'),  # First layer with 100 neurons
    Dense(50, activation='relu'),                       # Second layer with 50 neurons
    Dense(25, activation='relu'),                       # Third layer with 25 neurons
    Dense(10, activation='softmax')                     # Output layer with 10 neurons for classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.05),  # Randomly rotate images by up to 5%
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.05),      # Randomly zoom images by up to 5%
])

# Train the model with data augmentation
history = model.fit(data_augmentation(x_train), y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Save the trained model
model.save('larger_model_with_augmentation.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Plot loss history
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Model Evaluation
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_accuracy}')
print(f'Test loss: {test_loss}')

# Confusion Matrix
predictions = np.argmax(model.predict(x_test), axis=-1)
true_labels = np.argmax(y_test, axis=-1)
cm = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10))
plt.yticks(tick_marks, range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
