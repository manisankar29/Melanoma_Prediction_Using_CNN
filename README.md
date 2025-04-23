# Melanoma Detection Using CNN

This project is a web-based application that uses a Convolutional Neural Networks (CNN) to classify skin lesion images as **Benign** or **Malignant Melanoma**. It leverages a deep learning model trained on image data and integrates Google Cloud Storage (GCS) for model hosting.

## 🚀 Final Output

![Final Output](https://github.com/user-attachments/assets/1329775a-671a-4a60-97be-549483757286)

## 📌 Features

- Upload skin lesion image
- Model classifies as Benign or Malignant
- Uses CNN-based deep learning model
- Model loaded from Google Cloud Storage
- Displays input image and prediction result
- Web-based and responsive

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS
- **Backend**: Python, Flask
- **Model**: VGG16 model
- **Cloud**: Google Cloud Storage (GCS)
- **Deployment**: Render (Note: Free hosting platforms like **Render** or **Railway** may timeout on heavy model loading.)

## 📁 Project Structure

![Project Structure](https://github.com/user-attachments/assets/7c431b39-01d0-460d-b6e7-995fcd018cf5)

## ▶️ Run Locally

1. Clone the repository

```bash
git clone https://github.com/manisankar29/Melanoma_Prediction_Using_CNN.git
cd Melanoma_Prediction_Using_CNN
```

2. Set up virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Add your GCS key

Update the path in `app.py` with the correct location of your `gcs-key.json`

5. Run the app

```bash
python app.py
```
Open your browser and go to `http://127.0.0.1:5000`

## 🐳 Deployment Notes

- Model is large, so use cloud storage for hosting
- Free hosting platforms like **Render** or **Railway** may timeout on heavy model loading
- Consider using **Gunicorn** for production
- Use environment variables for secrets like GCS keys

## 📝 License

This project is licensed under the MIT License - see the [LICENSE]() file for details.
