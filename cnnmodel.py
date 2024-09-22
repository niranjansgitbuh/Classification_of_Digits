import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

"""Step 1: Load and preprocess the data"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

""" Step 2: Define the CNN model"""
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

""" Step 3: Compile the model"""
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

""" Step 4: Train the model"""
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

"""Step 5: Evaluate the model"""
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)
