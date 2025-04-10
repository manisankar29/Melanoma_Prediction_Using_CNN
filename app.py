import base64
from io import BytesIO
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from google.cloud import storage
from google.oauth2 import service_account
import io
import h5py
import os
import tempfile

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\manis\Downloads\gcs-key.json"

app = Flask(__name__)

BUCKET_NAME = "melanoma-model-storage"
MODEL_PATH = "skin.h5"
CREDENTIALS_PATH = r"C:\Users\manis\Downloads\gcs-key.json"

def load_model_from_gcs(bucket_name, model_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
        blob.download_to_file(temp_file)
        temp_file_path = temp_file.name

    model = load_model(temp_file_path)
    os.remove(temp_file_path)  # Optional cleanup
    return model

model = load_model_from_gcs(BUCKET_NAME, MODEL_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_bytes = BytesIO(file.read())  # Read file and convert to BytesIO
            img = image.load_img(img_bytes, target_size=(224, 224))  # Load image
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

            # Get the prediction
            prediction = model.predict(img_array)
            result = "Benign Melanoma Lesion" if prediction[0][0] > 0.5 else "Maligant Melanoma Lesion"

            # Convert image to Base64 for display
            img_bytes.seek(0)  # Reset cursor
            img_base64 = base64.b64encode(img_bytes.read()).decode()

            return render_template("result.html", image_data=img_base64, result=result)

    return render_template("index.html")

if __name__ == '__main__':
   app.run(debug=True)
