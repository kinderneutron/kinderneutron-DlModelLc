import os
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models

# Define paths to the dataset
dataset_dir = "C:/Users/hp/Downloads/project/human detection dataset"
person_dir = os.path.join(dataset_dir, "1")
no_person_dir = os.path.join(dataset_dir, "0")

# Function to preprocess images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))  # Resize image to match the input size of the model
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

# Load images and labels
images = []
labels = []

# Load images with person
for filename in os.listdir(person_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(person_dir, filename)
        images.append(preprocess_image(image_path))
        labels.append(1)  # Label for person presence

# Load images without person
for filename in os.listdir(no_person_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(no_person_dir, filename)
        images.append(preprocess_image(image_path))
        labels.append(0)  # Label for no person

# Convert lists to arrays
x = np.array(images)
y = np.array(labels)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Change to sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Change loss function for binary classification
              metrics=['accuracy'])

# Train the model
model.fit(x, y, epochs=10, batch_size=32)

# Now, let's use the trained model to detect if a person is present in real-time using the laptop camera
cap = cv2.VideoCapture(0)  # Open the default camera (0)

while True:
    ret, frame = cap.read()  # Read frame from camera
    if not ret:
        break

    # Preprocess the frame
    processed_frame = cv2.resize(frame, (100, 100)) / 255.0
    
    # Make prediction
    prediction = model.predict(np.expand_dims(processed_frame, axis=0))[0][0]
    
    # Display result
    if prediction > 0.5:
        cv2.putText(frame, "Person Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Person Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show the frame
    plt.imshow(frame)
    plt.title('Room Monitoring')
    plt.show()  


    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
