import os
import numpy as np
from PIL import Image
from numpy import asarray
from keras.models import load_model
from flask import Flask,request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)

model = load_model(r'models\weights_skin_disease_aug.h5')
model1 = load_model(r'models\weights_skin_cancer.h5')

model.make_predict_function()
model1.make_predict_function()
cls = {0: 'Actinic Keratosis', 1: 'Normal', 2: 'Skin Cancer'}


def model_predict(img_path, model):
    i = Image.open(img_path)
    i = i.resize((224, 224))
    i = asarray(i)
    i = i.reshape((1, 224, 224, 3))
    preds = model.predict(i)

    return preds


def model_predict1(img_path, model1):
    i = Image.open(img_path)
    i = i.resize((224, 224))
    i = asarray(i)
    i = i.reshape((1, 224, 224, 3))
    preds1 = model1.predict(i)

    return preds1


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        result = ""
        preds = model_predict(file_path, model)
        ypred = np.argmax(preds)
        preds1 = model_predict1(file_path, model1)

        if (ypred == 0):
            result = str(cls[ypred])
        elif (ypred == 1):
            result = str(cls[ypred])
        else:
            if(preds1>0.5):
                result = str(cls[ypred])+str(' ,Malignant')
            else:
                result = str(cls[ypred])+str(' ,Benign')


    return result



if __name__ == '__main__':
    app.run(debug=True)
