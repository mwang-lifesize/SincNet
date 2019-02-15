# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import sys
import os

import soundfile as sf
import numpy as np
#from conf import *
#from model import *

import numpy as np
import flask
import io

from predict_cpu import predict_model

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

user_mapping = np.load("/var/www/html/record/id_mapping.npy").tolist()

#np.random.seed(seed)

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    #model = predict_model("/home/mwang/Development/deep-learning/SincNet/exp/SincNet_lifesize/model_raw.pkl.lifesize")
    model = predict_model("/home/mwang/Development/deep-learning/SincNet/model_raw.pkl.lifesize")
    #model = predict_model("/home/mwang/Development/deep-learning/SincNet/exp/SincNet_libri/model_raw.pkl.amazon")
    #model = predict_model("/home/mwang/Development/deep-learning/SincNet/exp/SincNet_libri/model_raw.pkl.my_desktop")

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
            content = flask.request.json
            test_file = content["filename"]
            print("file name is:" + test_file)
            user, prob = model.predict(test_file)
            data["success"] = True
            data["user"] = user 
            data["prob"] = prob 

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
            "please wait until server has fully started"))
    load_model()
    #app.run( )
    # need this, otherwise error!!
    app.run( debug = False, threaded = False )
