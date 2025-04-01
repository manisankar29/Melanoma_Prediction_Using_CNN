import base64
from io import BytesIO
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

model = load_model('model/skin.h5')

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

# if __name__ == '__main__':
   # app.run(debug=True)
