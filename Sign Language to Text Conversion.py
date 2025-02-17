#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import os 
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from IPython.display import display
from PIL import Image  # Import Image from PIL


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# In[23]:


# ✅ Set dataset path (Update this if needed)
base_path = "E:/sign language to text/data2/train"  # Use relative path
if not os.path.exists(base_path):
    raise FileNotFoundError(f"Dataset folder {base_path} not found!")
labels = os.listdir(base_path)
print(labels)


# In[24]:


# ✅ Load labels (folders as classes)
labels = sorted(os.listdir(base_path))
num_classes = len(labels)
label_dict = {label: idx for idx, label in enumerate(labels)}


# In[25]:


# ✅ Load images and labels safely
x_data, y_data = [], []
num = []

valid_labels = []  # Store only labels that have images

for label in labels:
    path = os.path.join(base_path, label)
    if not os.path.exists(path):
        print(f"Skipping {label}, folder not found.")
        continue

    folder_data = os.listdir(path)
    k = 0
    print('\n', label.upper())

    for image_path in folder_data[:5]:  # Show only first 5 images
        image_full_path = os.path.join(path, image_path)
        if os.path.exists(image_full_path):
            display(Image.open(image_full_path))  # Open with PIL and display

    k = len(folder_data)
    num.append(k)
    print(f'There are {k} images in {label} class')

    for image_path in folder_data:
        image_full_path = os.path.join(path, image_path)
        image = cv2.imread(image_full_path)

        if image is None:
            print(f"Skipping corrupt image: {image_full_path}")
            continue

        image_resized = cv2.resize(image, (32, 32))  # Resize all images
        x_data.append(image_resized)
        y_data.append(label_dict[label])


# In[26]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
plt.bar(labels, num)
plt.title('NUMBER OF IMAGES CONTAINED IN EACH CLASS')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


# In[28]:


# ✅ Convert to NumPy arrays
X_train = np.array(x_data, dtype=np.float32) / 255.0  # Normalize images
Y_train = np.array(y_data, dtype=np.int32)

# ✅ One-hot encode labels
Y_train = to_categorical(Y_train, num_classes=num_classes)

# ✅ Check shapes before training
print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")


# In[29]:


x_data = np.array(x_data)
y_data = np.array(y_data)
print('the shape of X is: ', x_data.shape, 'and that of Y is: ', y_data.shape)


# In[30]:


x_data


# In[31]:


#stadardizing the input data
x_data = x_data.astype('float32')/255


# In[32]:


x_data


# In[33]:


#converting the y_data into categorical:
from sklearn.preprocessing import LabelEncoder
y_encoded = LabelEncoder().fit_transform(y_data)


# In[34]:


y_encoded


# In[35]:


from keras.utils import to_categorical
y_categorical = to_categorical(y_encoded)


# In[36]:


y_categorical


# In[37]:


#lets shuffle all the data we have:
r = np.arange(x_data.shape[0])
np.random.seed(42)
np.random.shuffle(r)
X = x_data[r]
Y = y_categorical[r]


# In[38]:


q = [1,2,3,4,5]
q


# In[39]:


np.random.shuffle(q)
q


# In[40]:


len(X),len(Y)


# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33)


# In[42]:


# ✅ Display some images
fig, axes = plt.subplots(3, 3, figsize=(6, 6))
fig.suptitle("Sample Images")
for i, ax in enumerate(axes.flat):
    if i < len(X_train):
        ax.imshow(X_train[i])
        ax.axis("off")
plt.show()


# In[43]:


# ✅ Define a CNN model
classifier = Sequential()
classifier.add(Conv2D(filters=32, kernel_size=3, padding="same", input_shape=(32, 32, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
classifier.add(Conv2D(filters=32, kernel_size=3, padding="same", activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
classifier.add(Conv2D(filters=64, kernel_size=3, padding="same", activation='relu'))
classifier.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
classifier.add(Flatten())
classifier.add(Dense(units=256, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=num_classes, activation='softmax'))


# In[46]:


# ✅ Compile the model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Ensure target and output have matching shapes before training
if Y_train.shape[1] != classifier.output_shape[1]:
    raise ValueError(f"Mismatch: Y_train.shape[1]={Y_train.shape[1]}, classifier.output_shape[1]={classifier.output_shape[1]}")


# In[47]:


# ✅ Train model
try:
    history = classifier.fit(X_train, Y_train, epochs=5, validation_split=0.2, batch_size=32)
except Exception as e:
    print(f"Error during training: {e}")
    exit()


# In[48]:


# ✅ Save the trained model
classifier.save("sign_language_model.h5")

# ✅ Model summary
classifier.summary()


# In[49]:


# ✅ Function for making predictions
def predict_sign(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Invalid image!")
        return
    image_resized = cv2.resize(image, (32, 32)) / 255.0
    image_array = np.expand_dims(image_resized, axis=0)
    prediction = classifier.predict(image_array)
    predicted_class = np.argmax(prediction)
    print(f"Predicted Sign: {labels[predicted_class]}")


# In[50]:


# Example Prediction
# predict_sign("./data/test/sign1.jpg")


# In[51]:


Y_train = Y_train.reshape(Y_train.shape[0], -1)  # Ensure shape (None, 28)
print(Y_train.shape)  # Should be (None, 28)

classifier.add(Dense(28, activation='softmax'))  # Ensure correct output size

history = classifier.fit(X_train, Y_train, epochs=10, validation_split=0.2)


# In[52]:


# Saving the model
classifier.save('my_model2.h5')

#print(classifier.history.keys())


# In[53]:


#Display of the accuracy and the loss values
import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss/accuracy')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[54]:


import numpy as np
from sklearn.metrics import classification_report

# Assuming you have imported and compiled your Keras Sequential model named 'classifier'
# and you have already loaded and preprocessed your data into X_test and Y_test

# Make predictions
Y_pred_prob = classifier.predict(X_test)  # Predicted probabilities for each class

# Convert predicted probabilities to class labels
Y_pred_classes = np.argmax(Y_pred_prob, axis=1)  # Convert probabilities to class labels

# Assuming Y_test contains true class labels
# Make sure Y_test is a 1D array or list of true class labels (not one-hot encoded)
# If Y_test is one-hot encoded, convert it to integer class labels
Y_test_classes = np.argmax(Y_test, axis=1)  # Convert one-hot encoded Y_test to integer class labels

print(classification_report(Y_test_classes, Y_pred_classes))


# In[55]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(np.argmax(y_true, axis=1), y_pred)
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, cmap='Purples')
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', 
                     color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.show()

# Make predictions
Y_pred = np.argmax(classifier.predict(X_test), axis=1)
plot_confusion_matrix(Y_test, Y_pred, labels)


# In[56]:


import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define your neural network model
classifer = Sequential()
classifer.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
classifer.add(MaxPooling2D(pool_size=(2, 2)))
classifer.add(Flatten())
classifer.add(Dense(units=24, activation='softmax'))

# Load and preprocess the test image
test_image = image.load_img("E:/sign language to text/data/train/A/1.jpg", target_size=(32, 32))
plt.imshow(test_image)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Make a prediction using the model
result = classifer.predict(test_image)

# Map the prediction to the corresponding label
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
prediction = labels[np.argmax(result)]

print(prediction)


# In[57]:


import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define your neural network model
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=24, activation='softmax'))

# Load and preprocess the test image
test_image = image.load_img("E:/sign language to text/data/train/A/1.jpg", target_size=(32, 32))
plt.imshow(test_image)
plt.show()  # You need to add this line to display the image
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Make a prediction using the model
result = classifier.predict(test_image)

# Map the prediction to the corresponding label
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
prediction = labels[np.argmax(result)]

print(prediction)


# In[58]:


get_ipython().system('pip install imutils')


# In[59]:


import keras
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

# Define the alphabet for sign language
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

# Load the pre-trained model
model = keras.models.load_model("my_model2.h5")

# Function to classify an image using the loaded model
def classify(image):
    image = cv2.resize(image, (32, 32))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    proba = model.predict(image)
    idx = np.argmax(proba)
    return alphabet[idx]

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 3)
    
    # Define the region of interest (ROI) for hand gesture
    top, right, bottom, left = 75, 350, 300, 590
    roi = img[top:bottom, right:left]
    roi = cv2.flip(roi, 1)
    
    # Preprocess the ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Classify the sign language gesture
    alpha = classify(gray)
    
    # Display the ROI rectangle and classification result on the frame
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, alpha, (0, 130), font, 5, (0, 0, 255), 2)
    
    # Display the processed frame
    cv2.imshow('img', img)
    
    # Exit the loop when 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




