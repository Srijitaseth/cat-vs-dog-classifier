from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('cats_vs_dogs_model.keras')

# Define a function to predict if an image is a cat or a dog
def predict_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(150, 150))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return "Dog"
    else:
        return "Cat"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = file.filename
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)
        result = predict_image(file_path)
        return render_template('result.html', filename=filename, result=result)

if __name__ == '__main__':
    app.run(debug=True)
