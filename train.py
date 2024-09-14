import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import cv2

# Preprocessing function to ensure images are properly centered and clean
def preprocess_image(image):
    # Resize the image to 28x28 (MNIST size)
    img_resized = cv2.resize(image, (28, 28))
    
    # Convert to grayscale if it's not already
    if len(img_resized.shape) == 3:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to clean up the image (helpful for poorly drawn '1's)
    _, img_thresholded = cv2.threshold(img_resized, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Normalize the image
    img_normalized = img_thresholded / 255.0

    return img_normalized.reshape(1, 28, 28, 1)  # Add batch dimension

# Function to train the model
def train_model(model_file):
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize the input data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Define the model architecture
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Output layer (10 classes)
    model.add(Dense(10, activation='softmax'))
    
    # Compile the model with a lower learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # Learning rate scheduler to reduce learning rate as training progresses
    def lr_scheduler(epoch, lr):
        return lr * 0.9 if epoch > 5 else lr
    
    lr_callback = LearningRateScheduler(lr_scheduler)
    
    # Train the model with validation split
    model.fit(X_train, y_train, epochs=20, validation_split=0.2, callbacks=[lr_callback])
    
    # Evaluate the model on the test data
    _, accuracy = model.evaluate(X_test, y_test)
    print(f'Model accuracy: {accuracy}')
    
    # Save the model to the specified file
    model.save(model_file)

# Predict function that ensures proper preprocessing
def predict_digit(model, drawn_image):
    # Preprocess the drawn image
    processed_img = preprocess_image(drawn_image)

    # Predict the digit
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)
    
    return predicted_class, prediction

# Entry point to train the model if executed directly
if __name__ == "__main__":
    model_file = 'model.keras'
    train_model(model_file)

