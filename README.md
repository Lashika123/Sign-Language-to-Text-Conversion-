# Sign Language to Text Conversion  

## Overview  
This project implements a deep learning-based model to convert sign language gestures into text. The model is trained on a dataset of sign language images and can predict the corresponding alphabet letters. It utilizes a Convolutional Neural Network (CNN) for image classification and supports real-time video-based prediction using OpenCV.  

## Features  
- Loads and preprocesses sign language dataset images  
- Trains a CNN model using TensorFlow/Keras  
- Supports real-time sign language recognition via webcam  
- Displays predictions on-screen with OpenCV  

## Technologies Used  
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- scikit-learn  

## Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/sign-language-text.git
   cd sign-language-text
   ```  

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

3. Ensure you have a dataset of sign language images stored in the correct directory (update `base_path` in the script).  

## Code Explanation  
### 1. Importing Necessary Libraries  
```python
import numpy as np
import os 
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
```
**Explanation:**  
- `numpy`, `os`, and `cv2` handle image processing and file management.  
- `tensorflow.keras` is used to build and train the CNN model.  

### 2. Loading Dataset  
```python
base_path = "E:/sign language to text/data2/train"
labels = os.listdir(base_path)
```
**Explanation:**  
- Defines the dataset location and loads labels from folder names.  

### 3. Preprocessing Images  
```python
x_data, y_data = [], []
for label in labels:
    path = os.path.join(base_path, label)
    for image_path in os.listdir(path):
        image = cv2.imread(os.path.join(path, image_path))
        image_resized = cv2.resize(image, (32, 32))
        x_data.append(image_resized)
        y_data.append(label_dict[label])
```
**Explanation:**  
- Reads and resizes images for training.  
- Converts labels into numerical values.  

### 4. Converting Data into NumPy Arrays  
```python
X_train = np.array(x_data, dtype=np.float32) / 255.0
Y_train = to_categorical(np.array(y_data), num_classes=num_classes)
```
**Explanation:**  
- Converts image data into NumPy arrays and normalizes pixel values.  
- One-hot encodes labels for multi-class classification.  

### 5. Building CNN Model  
```python
classifier = Sequential()
classifier.add(Conv2D(filters=32, kernel_size=3, padding="same", input_shape=(32, 32, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
classifier.add(Flatten())
classifier.add(Dense(units=256, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=num_classes, activation='softmax'))
```
**Explanation:**  
- Defines a CNN model with convolution, pooling, and dense layers for classification.  

### 6. Compiling and Training the Model  
```python
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, Y_train, epochs=5, batch_size=32, validation_split=0.2)
```
**Explanation:**  
- Uses `Adam` optimizer and categorical crossentropy loss for training.  
- Trains the model for 5 epochs.  

### 7. Saving the Model  
```python
classifier.save("sign_language_model.h5")
```
**Explanation:**  
- Saves the trained model for future use.  

### 8. Real-Time Prediction  
```python
def predict_sign(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (32, 32)) / 255.0
    image_array = np.expand_dims(image_resized, axis=0)
    prediction = classifier.predict(image_array)
    predicted_class = np.argmax(prediction)
    print(f"Predicted Sign: {labels[predicted_class]}")
```
**Explanation:**  
- Loads and preprocesses a new image.  
- Predicts the corresponding sign language alphabet.  

## Running Real-Time Sign Language Recognition  
To use the real-time sign detection feature via webcam, run:  
```bash
python real_time_recognition.py
```  
Press `q` to exit the webcam feed.  





  
  



