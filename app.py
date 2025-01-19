from flask import Flask, render_template, request
import os
import shutil
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODELS_FOLDER'] = 'models'

DEFAULT_COMIC = 'default'

def load_models_and_label_dict(comic_name):
    comic_folder = os.path.join(app.config['MODELS_FOLDER'], comic_name)
    if not os.path.exists(comic_folder):
        comic_folder = os.path.join(app.config['MODELS_FOLDER'], DEFAULT_COMIC)

    segmentation_model_path = os.path.join(comic_folder, 'segment.h5')
    segmentation_model = load_model(segmentation_model_path)

    classification_model_path = os.path.join(comic_folder, 'classify.h5')
    classification_model = load_model(classification_model_path)

    label_dict_path = os.path.join(comic_folder, 'label_dict.pkl')
    with open(label_dict_path, 'rb') as f:
        label_dict = pickle.load(f)

    return segmentation_model, classification_model, label_dict


def extract_faces(image, mask):
  mask = (mask > 0.5).astype(np.uint8) * 255
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  faces = []
  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    face = image[y:y+h, x:x+w]
    faces.append(face)

  return faces

def preprocess_face(face, target_size=(224, 224)):
    face = cv2.resize(face, target_size)
    face = face / 255.0
    return face


def classify_faces(faces, classify):
    predictions = []
    for face in faces:
        face = preprocess_face(face)
        face = np.expand_dims(face, axis=0)
        pred = classify.predict(face)
        pred_class = np.argmax(pred, axis=1)[0]
        predictions.append(pred_class)
    return predictions


def get_character_names(predictions, label_dict):
    return [label_dict[pred] for pred in predictions]


def clear_uploads_folder():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        clear_uploads_folder()

        files = request.files.getlist('file')
        comic_name = request.form['comic_name']

        segment, classify, label_dict = load_models_and_label_dict(comic_name)

        results = []
        for file in files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file_path = file_path.replace('\\', '/')
            file.save(file_path)

            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image, (512, 512))
            image_input = image_resized / 255.0
            image_input = np.expand_dims(image_input, axis=0)

            mask = segment.predict(image_input)[0]
            mask = np.squeeze(mask, axis=-1)

            faces = extract_faces(image_resized, mask)
            predictions = classify_faces(faces, classify)
            character_names = list(set([label_dict[pred] for pred in predictions]))

            results.append({
                'image_path': file_path,
                'character_names': character_names
            })

        return render_template('index.html', comic_name=comic_name, results=results)

    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)