# Classification_of_Digits
A Convolutional Neural Network (CNN) model built using TensorFlow/Keras to classify handwritten digits from the MNIST dataset. The model accurately predicts digits (0-9) by learning features from 28x28 grayscale images, showcasing effective deep learning for image recognition tasks.

# Solution
Using TensorFlow/Keras, we implement a CNN to learn features directly from the pixel values of the images. The model is trained on 60,000 images and evaluated on 10,000 test images. It consists of multiple layers, including convolutional layers for feature extraction and fully connected layers for classification.

# Key Features

- **Input:** 28x28 grayscale images of digits
- **Output:** Predicted digit (0-9)
- **Model:** Convolutional Neural Network with layers including Conv2D, Dense, and Dropout
- **Performance:** Achieves high accuracy on both the training and test sets
