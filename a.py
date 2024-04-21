import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# Define the larger model with more dense layers
model = Sequential([
    Dense(100, input_shape=(784,), activation='relu'),  # First layer with 100 neurons
    Dense(50, activation='relu'),                       # Second layer with 50 neurons
    Dense(25, activation='relu'),                       # Third layer with 25 neurons
    Dense(10, activation='softmax')                     # Output layer with 10 neurons for classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Save the trained model
model.save('larger_model.h5')