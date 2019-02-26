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
# used by ds
import wave
import flask
import io

from predict_cpu import predict_model

from deepspeech import Model, printVersions
from timeit import default_timer as timer

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
ds = None

user_mapping = np.load("/var/www/html/record/id_mapping.npy").tolist()
# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500
# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_ALPHA = 0.75
# The beta hyperparameter of the CTC decoder. Word insertion bonus.
LM_BETA = 1.85
# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training
# Number of MFCC features to use
N_FEATURES = 26
# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9

#np.random.seed(seed)

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    global ds
    #model = predict_model("/home/mwang/Development/deep-learning/SincNet/exp/SincNet_lifesize/model_raw.pkl.lifesize")
    model = predict_model("/home/mwang/Development/deep-learning/SincNet/model_raw.pkl.lifesize")
    #model = predict_model("/home/mwang/Development/deep-learning/SincNet/exp/SincNet_libri/model_raw.pkl.amazon")
    #model = predict_model("/home/mwang/Development/deep-learning/SincNet/exp/SincNet_libri/model_raw.pkl.my_desktop")
    ds_model = "/home/mwang/Development/deep-learning/stt/models/output_graph.pbmm"
    ds_alphabet = "/home/mwang/Development/deep-learning/stt/models/alphabet.txt"
    ds_lm = "/home/mwang/Development/deep-learning/stt/models/lm.binary"
    ds_trie = "/home/mwang/Development/deep-learning/stt/models/trie"
    ds = Model(ds_model, N_FEATURES, N_CONTEXT, ds_alphabet, BEAM_WIDTH)
    ds.enableDecoderWithLM(ds_alphabet, ds_lm, ds_trie, LM_ALPHA, LM_BETA)

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

            # deep speech
            fin = wave.open(test_file, 'rb')
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
            fs = fin.getframerate() # should be 16000
            text = ""
            if fs == 16000:
              text = ds.stt(audio, fs)
              print("text is:", text)
            else:
              # or do some conversion
              print("can not process as fs is not 16K")
            fin.close()

            data["success"] = True
            data["user"] = user 
            data["prob"] = prob 
            data["text"] = text 

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
