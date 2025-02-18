
# Sign Language to Text Conversion  
 
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

## Training the Model  
Run the following command to train the model:  
```bash
python Sign Language to Text Conversion.py
```  
This will preprocess the dataset, train the CNN model, and save the trained model as `sign_language_model.h5`.  

## Running Real-Time Sign Language Recognition  
To use the real-time sign detection feature via webcam, run:  
```bash
python real_time_recognition.py
```  
Press `q` to exit the webcam feed.  

## Example Usage  
To classify a new image:  
```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("sign_language_model.h5")

def predict_sign(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32)) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    print(f"Predicted Sign: {predicted_class}")

predict_sign("path_to_test_image.jpg")
```  



