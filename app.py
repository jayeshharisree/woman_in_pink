from datetime import timedelta
from flask import Flask, render_template,request,redirect, jsonify, abort
import flask_monitoringdashboard as dashboard
import requests_cache
import numpy as np
import db
import os
import time

#tf
import tensorflow as tf
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import efficientnet.tfkeras

#matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from werkzeug.utils import secure_filename

from utils.grad_cam import make_gradcam_heatmap

from model import InputForm
from compute import compute

import warnings
warnings.filterwarnings('ignore')

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#prediction dictionary
pred_dict = {0: 'No Cancer',
             1: 'Cancer'}

get_label = lambda lst: np.array([pred_dict[x] for x in lst])


expire_after = timedelta(minutes=30)
requests_cache.install_cache('main_cache',expire_after=expire_after)

app = Flask(__name__)
dashboard.bind(app)

# Model saved with Keras model.save()
MODEL_PATH = 'models/best_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model, datagen):
    img = image.load_img(img_path, target_size=(50, 50, 3))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)
    it = datagen.flow(x=x,
                      batch_size=1,
                      shuffle=False)
    preds = model.predict_generator(it)
    return preds

# make a prediction using test-time augmentation
def tta_prediction(datagen, model, x, steps=5):
  # prepare iterator

  it = datagen.flow(x=x,
                    batch_size=1,
                    shuffle=False)
  predictions = []
  for _ in range(steps):
      # make predictions for each augmented image
      yhats = model.predict(it, steps=it.n//it.batch_size, verbose=0)
      predictions.append(yhats)
  pred = np.mean(predictions, axis=0)
  return np.argmax(pred, axis=-1), np.max(pred), predictions

single_datagen = ImageDataGenerator(rescale=1/255.)

tta_datagen = ImageDataGenerator(
                rescale=1/255.,
                zoom_range=0.2,
                width_shift_range=0.2,
                height_shift_range=0.2)


@app.template_filter()
def numberFormat(value):
    return format(int(value), ',d')


@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html'), 404


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/faq')
def faq():
    return render_template("faq.html")


@app.route('/prevention')
def prevention():
    return render_template("prevention.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/diagnosis')
def diagnosis():
    return render_template("Diagnosis.html")


@app.route('/types')
def types():
    return render_template("types.html")


@app.route('/stages')
def stages():
    return render_template("stages.html")


@app.route('/facts')
def facts():
    return render_template("facts.html")


@app.route('/histopathology', methods=['GET'])
def detection():
    return render_template('detection.html')


@app.route('/v1/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)+"/static"
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        x_test = load_img(file_path, target_size=(50, 50, 3))
        x_test = img_to_array(x_test)  # this is a Numpy array with shape (3, 50, 50)
        x_test = preprocess_input(x_test)
        f, ax = plt.subplots(1, 1)

        # Generate class activation heatmap
        last_conv_layer_name = "top_activation"
        classifier_layer_names = [layer.name for layer in model.layers][-7:]#["flatten_2", "dense_2"]
        heatmap = make_gradcam_heatmap(x_test[np.newaxis]/255., model, last_conv_layer_name, classifier_layer_names)

        # We rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # We use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # We use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # We create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((x_test.shape[1], x_test.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * 0.4 + x_test
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

        ax.imshow(np.array(superimposed_img).astype("uint8"), interpolation='nearest')
        pred_class, pred_class_proba, _ = tta_prediction(tta_datagen, model, x_test[np.newaxis, :, :, :], steps=5)
        file_path = file_path.split('.')[0]+".png"
        time.sleep(0.5)
        f.savefig(file_path)
        
        print(pred_class_proba)
        
        
        no_cancer = f"{pred_dict[pred_class[0]]} with {pred_class_proba*100:.4f}% probability"
        cancer = f"{pred_dict[1-pred_class[0]]} with {(1-pred_class_proba)*100:.4f}% probability"
        return [no_cancer,cancer,file_path]
    return None


@app.route('/subscribed/<email>')
def subscribed(email):
    if(db.validateuser(email)):
        return render_template('subscribed.html')
    else:
        text='Email does not exist.'
        return render_template('error.html', text=text, again=True)


@app.route('/unsubscribe/<email>')
def unsubscribe(email):
    if(db.deluser(email)):
        return render_template('unsubscribed.html')
    else:
        text='You are not subscribed to this newsletter.'
        return render_template('error.html',text=text, again=False)


@app.route('/users')
def getusers():
    return str(db.getusers())


@app.route('/subscribe', methods=['POST'])
def subscribe():
    data = request.form
    email = data['email'].strip()

    if db.adduser(email):
        return render_template('index.html')
    else:
        text = 'User Already Exists'
        return render_template('index.html', text=text, again=True)

@app.route('/feature_based', methods=['GET', 'POST'])
def feature_based():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        result = compute(form.a.data, form.b.data, form.c.data, form.d.data, form.e.data, form.z.data, form.g.data,
                         form.h.data, form.i.data)

    else:
        result = None
    return render_template('feature.html', form=form, result=result)

# main driver function
if __name__ == '__main__':
    #app.run(host='0.0.0.0',port=5000,threaded=True)
    app.run(debug=True)
