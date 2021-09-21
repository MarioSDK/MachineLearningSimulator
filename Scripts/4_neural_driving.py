#parsing command line arguments
import argparse
#decoding camera images
import base64
#for frametimestamp saving
from datetime import datetime
#reading and writing files
import os
#high level file operations
import shutil
#matrix math
import numpy as np
#real-time server
import socketio
#concurrent networking 
import eventlet
#web server gateway interface
import eventlet.wsgi
#image manipulation
from PIL import Image
#web framework
from flask import Flask
#input output
from io import BytesIO
#load our saved model
from keras.models import load_model
#helper class
#import tanh_network
import utils
import cv2
import numpy as np
import glob
import math
# import imutils
import time

CLASS_LEFT     = "left"
CLASS_RIGHT    = "right"
CLASS_STRAIGHT = "straight"

CLASS_LEFT_25  = "left25"
CLASS_LEFT_20  = "left20"
CLASS_LEFT_15  = "left15"
CLASS_LEFT_10  = "left10"
CLASS_LEFT_05  = "left05"
CLASS_RIGHT_05 = "right05"
CLASS_RIGHT_10 = "right10"
CLASS_RIGHT_15 = "right15"
CLASS_RIGHT_20 = "right20"
CLASS_RIGHT_25 = "right25"

CLASS_LEFT3  = "left3"
CLASS_LEFT2  = "left2"
CLASS_LEFT1  = "left1"
CLASS_RIGHT1 = "right1"
CLASS_RIGHT2 = "right2"
CLASS_RIGHT3 = "right3"

ANGLES = {
	CLASS_LEFT     :-0.5,
	CLASS_RIGHT    : 0.5,
	CLASS_STRAIGHT : 0.0,
	CLASS_LEFT_25  :-1.0,
	CLASS_LEFT_20  :-0.8,
	CLASS_LEFT_15  :-0.6,
	CLASS_LEFT_10  :-0.4,
	CLASS_LEFT_05  :-0.2,
	CLASS_RIGHT_05 : 0.2,
	CLASS_RIGHT_10 : 0.4,
	CLASS_RIGHT_15 : 0.6,
	CLASS_RIGHT_20 : 0.8,
	CLASS_RIGHT_25 : 1.0,
	CLASS_LEFT3	   :-1.0,
	CLASS_LEFT2    :-0.6,
	CLASS_LEFT1	   :-0.3,
	CLASS_RIGHT1   : 0.3,
	CLASS_RIGHT2   : 0.6,
	CLASS_RIGHT3   : 1.0,
}

# set min/max speed for our autonomous car
MAX_SPEED = 20
MIN_SPEED = 10

# Pesos (Generados con R) de las entradas de las redes neuronales.
PESOS_TRES_LADOS  = np.genfromtxt('weights_three_sides.csv',  delimiter=',')
PESOS_SIETE_LADOS = np.genfromtxt('weights_seven_sides.csv',  delimiter=',')
PESOS_ONCE_LADOS  = np.genfromtxt('weights_eleven_sides.csv', delimiter=',')
PESOS_DE_ANGULOS  = np.genfromtxt('weights_angles.csv',       delimiter=',')

# Redes Neuronales de Tres y Once Angulos
RED_TRES_LADOS  = None
RED_SIETE_LADOS = None
RED_ONCE_LADOS  = None
RED_DE_ANGULOS  = None

# This class represents a Neural Network
class SimpleNeuralNetwork:
	
	def __init__(self, weights, layers, single_output=False, activation="sigmoid"):
		self.weights       = weights
		self.layers        = layers
		self.activation    = activation
		self.single_output = single_output
		self.output_names  = []
		for index in range(len(layers)):
			self.output_names.append("CLASS " + str(index))
	
	def set_output_names(self, names):
		self.output_names = names
	
	def normalize(self, value, limits=(-1, 1), target_limits=(0, 1)):
		limits = (float(limits[0]), float(limits[1]))
		target_limits = (float(target_limits[0]), float(target_limits[1]))
		value = float(value)

		size = limits[1] - limits[0]
		proportion = value - limits[0]
		other_size = target_limits[1] - target_limits[0]

		new_proportion = (proportion * other_size) / size
		return target_limits[0] + new_proportion

	def activation_sigmoid(self, input_value):
		negative_value = np.multiply(input_value, -1)
		output = 1 / (1 + np.exp(negative_value))
		return output
	
	def classify_image(self, any_image):
		crop_image  = np.array(any_image[57:136, 0:320])
		gray_image  = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
		array_image = np.asarray(gray_image)
		
		row    = np.array(np.arange(0, 80, 5),  dtype=np.intp)
		column = np.array(np.arange(0, 320, 5), dtype=np.intp)
		
		pixels = np.copy(array_image)
		pixels = pixels[row[:, np.newaxis], column]
		pixels = pixels.flatten()
		
		input_values = pixels
		pixel = 0
		
		for layer in range(1, len(self.layers)):
			input_size = self.layers[layer - 1] + 1
			outputs = []
			for neuron in range(self.layers[layer]):
				temporal_weights = self.weights[pixel:pixel + input_size]
				pixel += input_size
				weighted_sum = np.multiply(temporal_weights, np.transpose(np.append([1], input_values))).sum(axis=0)
				output = self.activation_sigmoid(weighted_sum)
				outputs.append(output)
			input_values = outputs
		
		if self.single_output:
			output = outputs[0]
			print("OUTPUT:     " + str(output))
			output = self.normalize(output, (0,1), (-1,1))
			print("NORMALIZED: " + str(output))
			print("++++++++++++++++++++++++++")
			return output * 5
		
		max_output = max(outputs)
		index = outputs.index(max_output)
		class_name = self.output_names[index]

		print("OUTPUTS:    " + str(outputs))
		print("MAX OUTPUT: " + str(max_output))
		print("INDEX:      " + str(index))
		print("CLASS NAME: " + class_name)
		print("+++++++++++++++++++++++++++++++")

		return class_name

############ -- INICIALIZACION -- ############

# Inicializacion de REDES NEURONALES:
RED_TRES_LADOS  = SimpleNeuralNetwork(PESOS_TRES_LADOS,  [1024, 5, 3, 3])
RED_SIETE_LADOS = SimpleNeuralNetwork(PESOS_SIETE_LADOS, [1024, 5, 3, 7])
RED_ONCE_LADOS  = SimpleNeuralNetwork(PESOS_ONCE_LADOS,  [1024, 5, 3, 11])
RED_DE_ANGULOS  = SimpleNeuralNetwork(PESOS_DE_ANGULOS,  [1024, 5, 3, 1], single_output=True)

# Nombres de las Clasificaciones/Salidas de las Redes:

RED_TRES_LADOS.set_output_names([
	CLASS_LEFT, CLASS_STRAIGHT, CLASS_RIGHT])

RED_SIETE_LADOS.set_output_names([
	CLASS_LEFT3, CLASS_LEFT2, CLASS_LEFT1, CLASS_STRAIGHT, 
	CLASS_RIGHT1, CLASS_RIGHT2, CLASS_RIGHT3])

RED_ONCE_LADOS.set_output_names([
	CLASS_LEFT_25, CLASS_LEFT_20, CLASS_LEFT_15, CLASS_LEFT_10, CLASS_LEFT_05, CLASS_STRAIGHT,
	CLASS_RIGHT_05, CLASS_RIGHT_10, CLASS_RIGHT_15, CLASS_RIGHT_20, CLASS_RIGHT_25])

# Inicializacion de SERVIDOR y APLICACION WEB:
sio = socketio.Server()
app = Flask(__name__)
#model = None
prev_image_array = None

# Variables Globales de Velocidad Maxima y Numero de Lados
speed_limit = MAX_SPEED
numero_de_lados = 3

#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)       # from PIL image to numpy array
            #image = utils.preprocess(image) # apply the preprocessing
            #image = np.array([image])       # the model expects 4D array

            # predict the steering angle for the image
            steering_angle = get_steering_angle(image)
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed == 0:
                speed = 1.5
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2
            if throttle < 0:
                throttle = 0.03
            print("STEERING ANGLE: " + str(steering_angle))
            print("THROTTLE:       " + str(throttle))
            print("SPEED:          " + str(speed))
            print("SPEED LIMIT:    " + str(speed_limit))
            print("+++++++++++++++++++++++++++++++++++")
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

    else:
        
        sio.emit('manual', data={}, skip_sid=True)

# Funcion de Conexion
@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

# Funcion para obtener el angulo de direccion
def get_steering_angle(any_image):
	if numero_de_lados == 1:
		return RED_DE_ANGULOS.classify_image(any_image)
	
	network = None
	if numero_de_lados == 3:
		network = RED_TRES_LADOS
	elif numero_de_lados == 7:
		network = RED_SIETE_LADOS
	elif numero_de_lados == 11:
		network = RED_ONCE_LADOS
	class_name = network.classify_image(any_image)
	return ANGLES[class_name]

# Funcion para enviar angulo y aceleracion al vehiculo
def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

########## -- FUNCIONES DE PRUEBA -- ###########

def test_image(filepath):
	image = cv2.imread(filepath)
	class_name = ""

	if numero_de_lados == 1:
		limit = 0.8
		angle = RED_DE_ANGULOS.classify_image(image)
		if angle < -limit:
			class_name = CLASS_LEFT
		elif (angle >= -limit) and (angle <= limit):
			class_name = CLASS_STRAIGHT
		elif angle > limit:
			class_name = CLASS_RIGHT
	else:
		network = None
		if numero_de_lados == 3:
			network = RED_TRES_LADOS
		elif numero_de_lados == 7:
			network = RED_SIETE_LADOS
		elif numero_de_lados == 11:
			network = RED_ONCE_LADOS
		class_name = network.classify_image(image)
		
	if class_name == CLASS_STRAIGHT:
		cv2.arrowedLine(image, (300, 120), (300, 30), (0, 255, 0), 5)
	elif class_name in [CLASS_LEFT, CLASS_LEFT1, CLASS_LEFT2, CLASS_LEFT3, CLASS_LEFT_05, CLASS_LEFT_10, CLASS_LEFT_15, CLASS_LEFT_20, CLASS_LEFT_25]:
		cv2.arrowedLine(image, (120, 30), (20, 30), (0, 255, 0), 5)
	elif class_name in [CLASS_RIGHT, CLASS_RIGHT1, CLASS_RIGHT2, CLASS_RIGHT3, CLASS_RIGHT_05, CLASS_RIGHT_10, CLASS_RIGHT_15, CLASS_RIGHT_20, CLASS_RIGHT_25]:
		cv2.arrowedLine(image, (120, 30), (280, 30), (0, 255, 0), 5)
		
	cv2.imshow('IMAGE CLASS: ' + class_name, image)
	cv2.waitKey(0)
	return class_name

def test_images(filepaths):
	classes = []
	for f_path in filepaths:
		classes.append(test_image(f_path))
	return classes

def test_folder(folderpath):
	filepaths = glob.glob(folderpath + "/*.jpg")
	return test_images(filepaths)

def make_angle_predictions():
	numero_de_lados = 3
	sides = [CLASS_LEFT, CLASS_STRAIGHT, CLASS_RIGHT]

	predictions = {}
	for real_side in sides:
		predictions["real_images_" + real_side] = {}
		for predicted in sides:
			predictions["real_images_" + real_side]["predicted_" + predicted] = 0
		classes = test_folder("images_three_sides_2/" + real_side)
		for predicted in classes:
			predictions["real_images_" + real_side]["predicted_" + predicted] += 1

	for real_side in sides:
		print("* Real images from " + real_side + " (Total = 400):")
		for predicted in sides:
			cantidad = predictions["real_images_" + real_side]["predicted_" + predicted]
			print("--- Predicted images from " + predicted + ": \t" + str(cantidad))

############# -- MAIN CODE -- #############

print("NOT RECORDING THIS RUN ...")

numero_de_lados = 3

#wrap Flask application with engineio's middleware
app = socketio.Middleware(sio, app)

#deploy as an eventlet WSGI server
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

#test_folder("images_three_sides/left")
#test_folder("images_three_sides/straight")
#test_folder("images_three_sides/right")
# make_angle_predictions()




