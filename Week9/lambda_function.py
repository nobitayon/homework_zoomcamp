# ## write as one
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor
import numpy as np
from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(x):
    x*= 1./255
    return x

interpreter = tflite.Interpreter(model_path = 'cats-dogs-v2.tflite')
interpreter.allocate_tensors()

# index which part of the model is the input
input_index = interpreter.get_input_details()[0]['index']
# index which part of the model is the output
output_index = interpreter.get_output_details()[0]['index']



def predict(url):
    img = download_image(url)
    X = np.array( prepare_image(img,target_size=(150,150)),dtype = 'float32')
    X = np.array([ preprocess_input(X) ])
    interpreter.set_tensor(input_index,X)
    # initialize input of interpreter with X

    interpreter.invoke()
    # we got the X as input and pass to the model and we have
    # result sitting in the output index

    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return float_predictions


def lambda_handler(event,context):
    # event is dictionary with parameters , we will have url parameter
    url = event['url']

    result = predict(url)

    return result


















