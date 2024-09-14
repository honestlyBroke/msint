import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

    # Data augmentation to improve model generalization
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False
    )
    
    datagen.fit(X_train.reshape(-1, 28, 28, 1))
    
    # Define the model architecture
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    # Output layer (10 classes)
    model.add(Dense(10, activation='softmax'))
    
    # Compile the model with a lower learning rate
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # Learning rate scheduler to reduce learning rate as training progresses
    def lr_scheduler(epoch, lr):
        if epoch > 15:
            return lr * 0.5
        elif epoch > 10:
            return lr * 0.75
        return lr
    
    lr_callback = LearningRateScheduler(lr_scheduler)
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model with validation split
    model.fit(datagen.flow(X_train.reshape(-1, 28, 28, 1), y_train, batch_size=64),
              epochs=50,
              validation_data=(X_test.reshape(-1, 28, 28, 1), y_test),
              callbacks=[lr_callback, early_stopping])
    
    # Evaluate the model on the test data
    _, accuracy = model.evaluate(X_test.reshape(-1, 28, 28, 1), y_test)
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
