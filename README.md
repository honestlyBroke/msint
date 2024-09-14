# Handwritten Digit Recognizer

This web application predicts the number that you have drawn on the canvas from 0-9 using a trained neural network model. It leverages the MNIST dataset and provides a simple interface for users to draw digits and see predictions.

## Features

- Draw digits on a canvas and get real-time predictions.
- Instructions provided for training your own model if the `model.keras` file does not exist.
- Visualize the input image and prediction results.

## Requirements

- Python 3.7+
- TensorFlow
- Streamlit
- OpenCV
- NumPy
- SciPy

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/honestlyBroke/msint.git
    cd msint
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. If the `model.keras` file is not present or if you want to train your own model, follow these steps:

    ### Training Your Own Model

    - Create a file named `train.py` in the same directory as `app.py`. You can use the provided `train.py` script or create your own with similar functionality.

    - Ensure the `train.py` script includes the necessary code to train a model using the MNIST dataset. Here is an example `train.py` script:

    ```python
    import os
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.callbacks import LearningRateScheduler
    from tensorflow.keras.optimizers import Adam

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
        
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Output layer (10 classes)
        model.add(Dense(10, activation='softmax'))
        
        # Compile the model with a lower learning rate
        optimizer = Adam(learning_rate=0.0001)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # Learning rate scheduler to reduce learning rate as training progresses
        def lr_scheduler(epoch, lr):
            return lr * 0.9 if epoch > 5 else lr
        
        lr_callback = LearningRateScheduler(lr_scheduler)
        
        # Train the model with validation split
        model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[lr_callback])
        
        # Evaluate the model on the test data
        _, accuracy = model.evaluate(X_test, y_test)
        print(f'Model accuracy: {accuracy}')
        
        # Save the model to the specified file
        model.save(model_file)

    if __name__ == "__main__":
        model_file = 'model.keras'
        train_model(model_file)
    ```

    - Execute the `train.py` script to train the model:

    ```bash
    python train.py
    ```

4. Once the model is trained and saved as `model.keras`, restart the Streamlit application to use the newly trained model.

## Deployed Application

You can also test the application online at the following URL: [Handwritten Digit Recognizer](https://msint-app.streamlit.app)

## File Structure

- `.gitignore`: Specifies files and directories to ignore in version control.
- `README.md`: This file.
- `app.py`: The main Streamlit application file.
- `requirements.txt`: List of required Python packages.
- `train.py`: Contains the function to train the model using the MNIST dataset.
- `model.keras`: The trained model file (generated after training).

## Possible Errors in Output

- **Incorrect Predictions**: The model might predict an incorrect digit due to unclear or ambiguous drawings.
- **Low Confidence**: For some drawings, the model's confidence in its prediction may be low, leading to less reliable results.
- **Canvas Issues**: The drawing on the canvas might not be correctly captured or processed, leading to errors in prediction.

## Future Plans

- **Model Improvement**: Train the model on a larger and more diverse dataset to improve accuracy.
- **Data Augmentation**: Implement data augmentation techniques to make the model more robust to variations in handwriting.
- **UI Enhancements**: Improve the user interface to provide better feedback and usability.
- **Real-time Feedback**: Provide real-time feedback and suggestions to users to help them draw clearer digits.
- **Deployment**: Deploy the application on a more robust platform to handle higher traffic and ensure better performance.

## Credits

- [Streamlit](https://www.streamlit.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)

Additionally, here is the `requirements.txt` file content for reference:

```
tensorflow
streamlit
opencv-python
numpy
streamlit-drawable-canvas
scipy
```