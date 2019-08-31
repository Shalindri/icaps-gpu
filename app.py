import json
import os

# import Marshmallow as Marshmallow
# import SQLAlchemy as SQLAlchemy
from _ast import List


import numpy as np
from biosppy import utils
from biosppy.signals import ecg
from flask import Flask, jsonify ,request
# from flask_marshmallow import Marshmallow
# from flask_sqlalchemy import SQLAlchemy
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.layers import LSTM
from numpy import newaxis
import h5py
import tensorflow as tf
from flask_cors import CORS, cross_origin
import json
from flask import render_template, request, redirect, url_for

import keras
import tensorflow as tf

from scipy import signal
# Init app
app = Flask(__name__)
CORS(app, resources=r'/api/*', allow_headers='Content-Type')

@app.route("/")
def helloWorld():
   return "Hello, cross-origin-world!"

basedir = os.path.abspath(os.path.dirname(__file__))
# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + \
    os.path.join(basedir, 'db.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config["IMAGE_UPLOADS"] = "/"
# Init db
# db = SQLAlchemy(app)
# # Init ma
# ma = Marshmallow(app)

# # Product Class/Model
# class Product(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(100), unique=True)
#     description = db.Column(db.String(200))
#     price = db.Column(db.Float)
#     qty = db.Column(db.Integer)

#     def __init__(self, name, description, price, qty):
#         self.name = name
#         self.description = description
#         self.price = price
#         self.qty = qty

# # Product Schema


# class ProductSchema(ma.Schema):
#     class Meta:
#         fields = ('id', 'name', 'description', 'price', 'qty')


# # Init schema
# product_schema = ProductSchema(strict=True)
# products_schema = ProductSchema(many=True, strict=True)

# # Create a Product
# @app.route('/product', methods=['POST'])
# def add_product():
#     name = request.json['name']
#     description = request.json['description']
#     price = request.json['price']
#     qty = request.json['qty']

#     new_product = Product(name, description, price, qty)

#     db.session.add(new_product)
#     db.session.commit()

#     return product_schema.jsonify(new_product)

# # Get All Products
# @app.route('/product', methods=['GET'])
# def get_products():
#     all_products = Product.query.all()
#     result = products_schema.dump(all_products)
#     return jsonify(result.data)

# # Get Single Products
# @app.route('/product/<id>', methods=['GET'])
# def get_product(id):
#     product = Product.query.get(id)
#     return product_schema.jsonify(product)

# # Update a Product
# @app.route('/product/<id>', methods=['PUT'])
# def update_product(id):
#     product = Product.query.get(id)

#     name = request.json['name']
#     description = request.json['description']
#     price = request.json['price']
#     qty = request.json['qty']

#     product.name = name
#     product.description = description
#     product.price = price
#     product.qty = qty

#     db.session.commit()

#     return product_schema.jsonify(product)

# # Delete Product
# @app.route('/product/<id>', methods=['DELETE'])
# def delete_product(id):
#     product = Product.query.get(id)
#     db.session.delete(product)
#     db.session.commit()

# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1})
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)


#     return product_schema.jsonify(product)
def load_model():
    ### Loading a Check-Pointed Neural Network Model
    # How to load and use weights from a checkpoint
    best_model_file = "./model_trained.hdf5"
    ### Loading a Check-Pointed Neural Network Model
    # How to load and use weights from a checkpoint
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import ModelCheckpoint
    import matplotlib.pyplot as plt

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # create model
    print('Build LSTM RNN model ...')
    model = Sequential()
    model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=(256, 1)))
    model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    model.add(Dense(15, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse', 'mae', 'mape', 'cosine'])
    model.summary()
    # load weights
    model.load_weights(best_model_file)
    # Compile model (required to make predictions)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Created model and loaded weights from file")

    return model


def load_txt(path):

    # normalize path
    path = utils.normpath(path)

    # read file line by line
    with open(path, 'rb') as fid:
        lines = fid.readlines()

    values = []   # values in the input file
    for item in lines:
        values.append(item)

    sampling_rate = 256
    resolution = 12

    # convert mdata
    mdata = {}   # a dictionary to hold the values
    df = '%Y-%m-%dT%H:%M:%S.%f'
    try:
        mdata['sampling_rate'] = float(sampling_rate)
    except KeyError:
        pass
    try:
        mdata['resolution'] = int(resolution)
    except KeyError:
        pass

    # load array
    data = np.genfromtxt(values, delimiter=b',')

    return data, mdata

@app.route("/upload", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":

        if request.files:
            image = request.files["image"]

            print(image)

            return redirect(request.url)
    return 'uploaded'



@app.route('/api/prediction/<filePath>', methods=['GET'])
@cross_origin()
def predict_arrhythmia(filePath):
    signal2 = load_txt('./' + filePath)
    print('signal', signal2)

    channel = signal2[0].transpose()
    print('data = record[0].transpose():', channel)

    # Find rpeaks in the ECG data. Most should match with the annotations.
    out = ecg.ecg(signal=channel, sampling_rate=360, show=False)
    # print('r peaks', out[2])
    # create a array size similar to signal array ex:-650000 with all zero values
    rpeaks = np.zeros_like(channel, dtype='float')
    # replace '1.0' for indexes in the array comparing with the -out['rpeaks']- array
    rpeaks[out['rpeaks']] = 1.0

    beatstoremove = np.array([0])
    beatstoremove2 = np.array([0])

    # Split into individual heartbeats. For each heartbeat
    # record, append classification (normal/abnormal).
    beats = np.split(channel, out['rpeaks'])
    br = beats.copy()

    for idx, idxval in enumerate(out['rpeaks']):

        firstround = idx == 0
        lastround = idx == len(beats) - 1

        # Skip first and last beat.
        if (firstround or lastround):
            continue

        #overlap_readings = 512 - beats[idx].size
        beats[idx] = np.append(beats[idx], beats[idx+1][:100])
        br[idx] = np.append(br[idx], br[idx + 1][:100])

        # Normalize the readings to a 0-1 range for ML purposes.
        beats[idx] = (beats[idx] - beats[idx].min()) / beats[idx].ptp()
        print(len(beats[idx]))



        # Resample from 360Hz to 125Hz
        #newsize = int((beats[idx].size * 125 / 360) + 0.5)
        beats[idx] = signal.resample(beats[idx], 256)
        br[idx] = signal.resample(br[idx], 256)
        # print(len(beats[idx]),len(br[idx]))

        if (beats[idx].size > 256):
            beatstoremove = np.append(beatstoremove, idx)
            continue
        if (br[idx].size > 256):
            beatstoremove2 = np.append(beatstoremove2, idx)
            continue

        # Append the classification to the beat data.
       # beats[idx] = np.append(beats[idx], catval)

    beatstoremove = np.append(beatstoremove, len(beats)-1)
    beatstoremove2 = np.append(beatstoremove2, len(br) - 1)

    # Remove first and last beats and the ones without classification.
    beats = np.delete(beats, beatstoremove)
    br = np.delete(br, beatstoremove2)

    beatdata = np.array(list(beats[:]))
    beatdata2 = np.array(list(br[:]))
    # print(len(beats))
    beat_array = beatdata.reshape((len(beats), 256, 1))
    beat_array2 = beatdata.reshape((len(br), 256, 1))
    print(beat_array.shape)
    print(beat_array2.shape)


    ### Loading a Check-Pointed Neural Network Model
    # How to load and use weights from a checkpoint
    best_model_file = "./model_trained.hdf5"
    ### Loading a Check-Pointed Neural Network Model
    # How to load and use weights from a checkpoint
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import ModelCheckpoint
    import matplotlib.pyplot as plt

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # create model
    print('Build LSTM RNN model ...')
    model = Sequential()
    model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=(256, 1)))
    model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    model.add(Dense(15, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse', 'mae', 'mape', 'cosine'])
    model.summary()
    # load weights
    model.load_weights(best_model_file)
    # Compile model (required to make predictions)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Created model and loaded weights from file")

    y_pred = model.predict_classes(beat_array, batch_size=32)
    # print(y_pred)
    json_list = []
    for id, x in enumerate(beat_array):
        y =br[id].transpose().tolist()
        myList = [round(z) for z in y]
        pred_list =[]
        for catid, catval in enumerate(y_pred):
            if (catval == 0):
               catval = "Normal"
               pred_list.append(catval)
            elif (catval == 1):
                catval = "Left Bundle Branch Block"
                pred_list.append(catval)
            elif (catval == 2):
                catval = "Right Bundle Branch Block"
                pred_list.append(catval)
            elif (catval == 3):
                catval = "Atrial Premature Beat"
                pred_list.append(catval)
            elif (catval == 4):
                catval = "Aberrated Atrial Premature Beat"
                pred_list.append(catval)
            elif (catval == 5):
                catval = "Nodal (junctional)Premature Beat"
                pred_list.append(catval)
            elif (catval == 6):
                catval = "x"
                pred_list.append(catval)
            elif (catval == 7):
                catval = "PVC"
                pred_list.append(catval)
            elif (catval == 8):
                catval = "x"
                pred_list.append(catval)
            elif (catval == 9):
                catval = "x"
                pred_list.append(catval)
            elif (catval == 10):
                catval = "x"
                pred_list.append(catval)
            elif (catval == 11):
                catval = "x"
                pred_list.append(catval)
            elif (catval == 12):
                catval = "x"
                pred_list.append(catval)
            elif (catval == 13):
                catval = "x"
                pred_list.append(catval)
            elif (catval == 14):
                catval = "x"
                pred_list.append(catval)

        beat_obj = ({"beats": list(myList),"predicted_class": str(pred_list[id]),"id":str(id )})
        json_list.append(beat_obj)



    # y_pred = model.predict_classes(beats_mat, batch_size=32)
    # print(list(json_list))
    json_list = list(json_list)
    complete_beat = list(channel)
    x = "arrhythmia "
    obj = {"key": "jesus", "predictionArray":"itiy","prediction": ((json_list))}
    return (json.dumps(obj),200, {'content-type': 'application/json'})
    e = time.time()
    print(e - s)

# Run Server
if __name__ == '__main__':
    app.run(debug=True)
