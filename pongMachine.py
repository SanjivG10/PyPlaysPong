from PIL import ImageGrab
import numpy as np
import cv2
import keyboard
import pickle
import random

#image grabbing constants
OFFSET_WIDTH = 150
OFFSET_HEIGHT = 160

WIDTH =640+OFFSET_WIDTH
HEIGHT=480+OFFSET_HEIGHT


UP_ARROW = np.array([1])
DOWN_ARROW = np.array([0])

reduced_width = WIDTH//12
reduced_height = HEIGHT//12


# train_data = []
# def train():
#     while True:
#         img = ImageGrab.grab(bbox=(10,40,WIDTH,HEIGHT))
#         img_np = np.array(img) #this is the array obtained from conversion
#         frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
#         cv2.imshow("test", frame)
#         reduced_frame = cv2.resize(frame,(reduced_width,reduced_height))
#         key = None
#         try:
#             if keyboard.is_pressed('up'):
#                 key = UP_ARROW
#                 train_data.append([reduced_frame,key])
#             elif keyboard.is_pressed('down'):
#                 key = DOWN_ARROW
#                 train_data.append([reduced_frame,key])
#         except Exception as e:
#             print(e)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#                 cv2.destroyAllWindows()
#                 break
# train()
#
# with open('pong_train2.pickle','wb') as pickle_file:
#     pickle.dump(train_data,pickle_file)
#     pickle_file.close()


with open('pong_train2.pickle','rb') as pickle_file:
    data = pickle.load(pickle_file)
    for  i in range(100):
        random.shuffle(data)
    #normalized dataset
    X= np.array( [ i[0]  for i in data ])/225.0
    y = [ i[1] for i in data]

def requiredParam():
    return X,y
