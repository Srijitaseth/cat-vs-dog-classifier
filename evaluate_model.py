# evaluate_model.py

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
import numpy as np
import os

# Load the model
model = tf.keras.models.load_model('cats_vs_dogs_model.keras')

# Define the test directory
test_dir = 'cats_vs_dogs/test'

# Data augmentation and rescaling for validation
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Flow test images in batches of 32 using test_datagen generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Make sure the order is the same for the predictions
)

# Get the ground truth labels
y_true = test_generator.classes

# Predict labels for the test set
y_pred_probs = model.predict(test_generator)
y_pred = (y_pred_probs > 0.5).astype("int32")

# Analyze the Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cat', 'Dog'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save the confusion matrix plot
plt.show()

# Print the classification report
report = classification_report(y_true, y_pred, target_names=['Cat', 'Dog'])
print("Classification Report:")
print(report)

# Load the history object (Make sure you saved the history object as a pickle file in your training script)
import pickle

with open('training_history.pkl', 'rb') as file:
    history = pickle.load(file)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')  # Save the plot

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')  # Save the plot

plt.tight_layout()
plt.show()
