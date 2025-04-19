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
import json

app = Flask(__name__)

# Cloud Storage setup
BUCKET_NAME = "melanoma-model-storage"
MODEL_PATH = "skin.h5"

# Load GCS credentials from environment variable (Render-safe)
gcs_key_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if gcs_key_json:
    credentials_dict = json.loads(gcs_key_json)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    client = storage.Client(credentials=credentials)
else:
    raise ValueError("Google Cloud credentials not found.")

def load_model_from_gcs(bucket_name, model_path, client):
    try:
        print("Connecting to GCS...")
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(model_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
            blob.download_to_file(temp_file)
            temp_file_path = temp_file.name
            print(f"Model downloaded to: {temp_file_path}")

        model = load_model(temp_file_path)
        print("Model loaded successfully!")
        os.remove(temp_file_path)  
        return model

    except Exception as e:
        print("Failed to load model from GCS:", e)
        raise

# Load model on startup
model = load_model_from_gcs(BUCKET_NAME, MODEL_PATH, client)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_bytes = BytesIO(file.read())
            img = image.load_img(img_bytes, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)
            result = "Benign Melanoma Lesion" if prediction[0][0] > 0.5 else "Maligant Melanoma Lesion"

            img_bytes.seek(0)
            img_base64 = base64.b64encode(img_bytes.read()).decode()

            return render_template("result.html", image_data=img_base64, result=result)

    return render_template("index.html")

#if __name__ == '__main__':
   # import os
   # port = int(os.environ.get("PORT", 10000))  
   # app.run(host='0.0.0.0', port=port)
