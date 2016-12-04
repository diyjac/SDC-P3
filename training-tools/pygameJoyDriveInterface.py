##
## This is an initial proof-of-concept to interface with the drive.py interface with the Unity SDC simulator
## Original idea was to record the test data as if it came from the simulator, but that's nothing more than
## just doing it directly from the simulator.  There has to be a better way to train...
##

import argparse
import base64
import json
import cv2
import pygame

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

### initialize pygame and joystick
pygame.init()
pygame.joystick.init()

ch, img_rows, img_cols = 3, 66, 200  # Nvidia's camera format
def preprocess(image):
    image = cv2.resize(image, (img_cols, img_rows), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_prep = np.asarray(image)
    image_array = preprocess(image_prep)

    recording = False
    for event in pygame.event.get():
        if event.type == pygame.JOYAXISMOTION:
            print("Joystick moved")
        if event.type == pygame.JOYBUTTONDOWN:
            print("Joystick button pressed.")

    # Get joystick and initialize
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    # We are using PS4 left joystick: so axis (0,1) run in pairs, left/right for 2, up/down for 3
    # Normally they are centered on (0,0)
    leftright = joystick.get_axis(0)/2.0
    updown = joystick.get_axis(1)
    if leftright < -0.01 or leftright > 0.01:
        if joystick.get_button(0) == 0:
            recording = True
    if recording:
        print("Recording: ")
        print("Right Stick Left|Right Axis value {:>6.3f}".format(leftright) )
        print("Right Stick Up|Down Axis value {:>6.3f}".format(updown) )

    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1)) + leftright
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.5 + updown
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
