import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import ImageGrab
import numpy as np
import cv2
from pynput.keyboard import Controller
import time
from keycontroller import PressKey,ReleaseKey,W,S


OFFSET_WIDTH = 150
OFFSET_HEIGHT = 160

WIDTH =640+OFFSET_WIDTH
HEIGHT=480+OFFSET_HEIGHT



model = load_model('my_model.h5')

keyboard = Controller()
while True:
    img = ImageGrab.grab(bbox=(10,40,WIDTH,HEIGHT))
    img_np = np.array(img) #this is the array obtained from conversion
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    cv2.imshow("test", frame)

    inputFrame = cv2.resize(frame,(65,53))
    cv2.imshow("test", inputFrame)
    inputFrame = np.reshape(inputFrame,(-1,65,53,1))
    inputFrame = inputFrame/255.0
    output =  np.around(model.predict(inputFrame))
    print (output)
    if output==1:
        keyToPress = W
    else:
        keyToPress= S
    try:
        PressKey(keyToPress)
        time.sleep(2)
        ReleaseKey(keyToPress)
        print ("PRESSING {}".format(keyToPress))
    except Exception as e:
        print(e)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
