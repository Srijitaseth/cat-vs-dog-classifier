import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('cats_vs_dogs_model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return "Dog"
    else:
        return "Cat"

# Example usage
print(predict_image('cats_vs_dogs/test/cat.jpg'))
