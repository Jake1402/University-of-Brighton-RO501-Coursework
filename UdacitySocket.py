from collections import deque

import socketio
import eventlet
import eventlet.wsgi
from flask import Flask

import base64

import os
import time
import cv2, numpy as np
import torch
from torchvision.transforms import v2

from EffecientRes import ConvolutionalAgent
from PID import PIDCon

"""
Lets the user view the scene that model sees
just in the RGB format instead of YUV.
"""
frame_buffer = deque(maxlen=32)
def cvLoop():
    print("Starting CV Window.")
    while True:
        if len(frame_buffer) > 0:
            frame = frame_buffer.popleft()  # FIFO rules!
            frame = frame[60:135, :]
            upsize = cv2.resize(frame , (320, 160))
            cv2.imshow("Agent View", upsize)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            eventlet.sleep(0.01)
    cv2.destroyAllWindows()

"""
This function builds the model, initialises params and finally
loads the PT file, before returning it.
"""
def spawn_model(path):
    Agent = ConvolutionalAgent(action_space=1)
    Agent.forward(torch.zeros([1,3,160,80]))
    state_dict = torch.load(path, weights_only=True, map_location="cpu")
    Agent.load_state_dict(state_dict)
    Agent.eval()
    return Agent

if __name__ == "__main__":

    PGain = 1.2     # Proportional Gain
    IGain = 0.15    # Integral Gain
    DGain = 0.1     # Derivative Gain
    MaxI = +5       # Upper limit of Integral (Prevents blow ups)
    LowI = -5       # Lower limit of Integral (Prevents blow ups)

    PATH = os.path.abspath("Models/Model at epoch - 20.pt")  # Model to use
    device = "cuda" if torch.cuda.is_available() else "cpu"   # Use GPU if possible
    device = "cpu"
    Agent = spawn_model(PATH).to(device)                      # Spawn model
    basic_transform = v2.Compose([                            # Define transforms
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    ThrottleController = PIDCon(PGain, IGain, DGain, MaxI, LowI)

    term_time = time.time()
    frame_list = []

    sio = socketio.Server()
    app = Flask(__name__)
    # Retrieve telemetry from socket
    @sio.on('telemetry')
    def telemetry(sid, data):
        if data:
            init_time = float(time.time())
            img_b64 = data['image']                                 # Grab base64 image
            img_bytes = base64.b64decode(img_b64)                   # Decode from base64 format
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)    # Output bytes into a 1D np array
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)       # Convert np byte array to image
            frame_buffer.append(frame)                              # Send base64 image to buffer

            img = np.array(frame)[60:135, :]                        # Crop the top portion of image (No useful info)
            frame = cv2.resize(img, (160, 80))                # Resize image for model.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)          # Convert to YUV for model
            np_frame = np.reshape(np.array(frame, dtype=np.float32), (3, 160, 80))/255.0 # Reshape channel first
            #print(torch.max(basic_transform(torch.from_numpy(np_frame).unsqueeze(0))))
            # Get steering and throttle values
            steer_prediction = Agent(basic_transform(torch.from_numpy(np_frame).unsqueeze(0).to(device)))
            throttle = ThrottleController.control(
                target=30,
                current=float(data["speed"]),
                dt=float(time.time() - init_time)
            )

            # Send neutral control back (required or sim stalls)
            send(min(throttle/30, 1.0), steer_prediction.item())
        else:
            prev_theta = 0
            sio.emit('manual', data={}, skip_sid=True)

    # Function to send commands to server
    def send(throttle, steering):
        sio.emit(
            "steer",
            data = {
                "steering_angle": str(steering),    #   MUST BE A STRING OR WONT FUNCTION
                "throttle": str(throttle)           #   MUST BE A STRING OR WONT FUNCTION
            }, skip_sid=True
        )

    # Inform I have connected to socket
    @sio.on('connect')
    def connect(sid, environ):
        print("Simulator connected:", sid)

    # Inform I have disconnected from socket.
    @sio.on('disconnect')
    def disconnect(sid):
        print("Simulator Disconnected:", sid)

    # Begins thread for server communications
    eventlet.spawn(cvLoop)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(
        eventlet.listen(('', 4567)),
        app
    )




