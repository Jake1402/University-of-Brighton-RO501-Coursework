import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os, cv2

"""
https://stackoverflow.com/questions/14313510/how-can-i-calculate-a-rolling-moving-average-using-python-numpy-scipy
Moving average code used from from this github repo, unfortunatly it shrinks the size proportionally to the 
moving size. To remedy this in the DatasetLoader code I trim both the X and y components down to the shortest.
movingAvgN(np.array(df.iloc[:,3]), 1)
"""
def movingAvgN(arr, n):
    print(f"Shape before - {arr.shape}")
    arr = np.concat([arr[:n-1], arr])
    ret = np.cumsum(arr, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    print(f"Shape before - {ret.shape}")
    return ret

def returnImageArray(arr, HSV=True):
    img_buffer = []
    for image_loc in arr[::2]:
        img = cv2.imread(image_loc)
        img = np.array(img)[60:135, :]
        img = cv2.resize(img, (160, 80))
        if HSV:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_array = np.array(img, dtype=np.float32)/255.0
        channel_first = np.reshape(img_array, (3, 160, 80))
        img_buffer.append(channel_first)
    return np.array(img_buffer)

"""
V2 Version is significantly faster. Uses numpys multithreading to do
all array reshaping and multiplication in parallel instead of
sequentially. 
"""
def returnImageArrayV2(arr, HSL=True):
    img_buffer = []
    for image_loc in arr[::2]:
        img = cv2.imread(image_loc)[60:135, :]
        img = cv2.resize(img, (160, 80))
        if HSL:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_buffer.append(img)
    img = np.array(img_buffer)
    img_array = np.array(img, dtype=np.float16)/255.0
    img_buffer = np.reshape(img_array, (img_array.shape[0], 3, 160, 80))
    return img_buffer

def save(zippedXY, path : str):
    print(f"Saving to device at location: {path}")
    zipped = np.asarray(zippedXY, dtype=np.float32)
    np.save(path, zipped)
    print("Saving complete")

if __name__ == "__main__":
    PATH = "D:\\Udacity Simulator\\controller\\"

    # Data in format C_img, L_img, R_img, Steer, Accel, Deccel, Velocity
    df = pd.read_csv(PATH+"driving_log.csv")
    center, left, right = np.array(df.iloc[:,0]), np.array(df.iloc[:,1]), np.array(df.iloc[:,2])

    # Smooths the inputs to remove any sudden jerks
    print("Loading & Saving Steer Angles")
    steer_angle = np.array(df.iloc[:,3])
    steer_angle = movingAvgN(steer_angle, 3)
    save(steer_angle, os.path.abspath("npysets\\steer"))
    #steer_angle = np.clip(steer_angle, a_min=-(10.0/25.0), a_max=(10.0/25.0))

    print("Loading & Saving Images")
    print("Center Images")
    center_buffer = returnImageArrayV2(center)
    save(center_buffer, os.path.abspath("npysets\\center"))
    del center_buffer

    print("Left   Images")
    left_buffer   = returnImageArrayV2(left)
    save(left_buffer, os.path.abspath("npysets\\left"))
    del left_buffer

    print("Right  Images")
    right_buffer  = returnImageArrayV2(right)
    save(right_buffer, os.path.abspath("npysets\\right"))
    del right_buffer

    plt.plot(steer_angle)
    plt.show()
    del steer_angle

