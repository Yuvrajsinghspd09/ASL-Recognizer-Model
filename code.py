import os
from PIL import Image
import matplotlib.pyplot as plt

# Path to the train folder containing images
#train_folder = 'ASL_FOLDER\American Sign Language Letters.v1-v1.tensorflow\train'

import os

train_folder = 'ASL_FOLDER\\American Sign Language Letters.v1-v1.tensorflow\\train'

if os.path.exists(train_folder):
    print("Directory exists")
else:
    print("Directory does not exist or path is incorrect")

# Count instances of each alphabet class
class_counts = {}

# Loop through the images in the train folder
displayed_images = 0
for filename in os.listdir(train_folder):
    if displayed_images >= 10:
        break  # Exit loop after displaying 10 images
    
    # Open the image using PIL
    image_path = os.path.join(train_folder, filename)
    image = Image.open(image_path)
    
    # Extract class label from the first character of the filename
    class_label = filename[0].upper()  # Assuming the first character denotes the class label
    
    # Display image and class label
    plt.imshow(image)
    plt.title(f"Class: {class_label}, Image: {filename}")
    plt.show()
    
    # Count instances for each class
    if class_label not in class_counts:
        class_counts[class_label] = 0
    class_counts[class_label] += 1
    
    displayed_images += 1

# Display the count of instances for each class
print(class_counts)


import cv2
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    labels = []
    
    for filename in os.listdir(folder):
        try:
            image_path = os.path.join(folder, filename)
            
            # Check if file is an image
            if not (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
                continue
            
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Resize image to 200x200
            image = cv2.resize(image, (200, 200))
            
            # Normalize pixel values between 0 and 1
            image = image.astype('float32') / 255.0
            
            images.append(image)
            
            label = ord(filename[0]) - ord('A')
            labels.append(label)
        except Exception as e:
            print(f"Error loading image: {e}")
    
    return np.array(images), np.array(labels)

train_folder = 'ASL_FOLDER\American Sign Language Letters.v1-v1.tensorflow\\train'
test_folder = 'ASL_FOLDER\American Sign Language Letters.v1-v1.tensorflow\\test'

train_images, train_labels = load_images_from_folder(train_folder)
test_images, test_labels = load_images_from_folder(test_folder)

# Reshape images for CNN input
train_images = train_images.reshape(-1, 200, 200, 1)
test_images = test_images.reshape(-1, 200, 200, 1)

# Display shapes for verification
print("Train images shape:", train_images.shape)
print("Test images shape:", test_images.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Build a simple CNN model without data augmentation, regularization, and dropout
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with the original training data
history = model.fit(train_images, train_labels, epochs=3, validation_data=(test_images, test_labels))

import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 200))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

def predict_images(image_paths, model):
    for image_path in image_paths:
        img = preprocess_image(image_path)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        print(f"Image: {image_path}, Predicted Class: {chr(predicted_class + ord('A'))}")

# Load your image data
image_paths = [
    'download (1).jpg',
    'download.jpg',
    'maxresdefault.jpg'
]

# Call the function to predict images
predict_images(image_paths, model)


# Define the file paths to save the model and its weights
model_path = 'saveddd'
weights_path = 'saveddd'

# Save the model architecture and weights
model.save(model_path)  # Saves the model architecture to a HDF5 file
model.save_weights(weights_path)  # Saves the model weights to a HDF5 file
