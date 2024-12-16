import cv2 as cv 
import numpy as np 
import time 
 
import pycoral.utils.edgetpu as edgetpu 
from pycoral.adapters import common, classify 
 
import tkinter as tk 
 
# Servo 
from gpiozero import Servo 
import math 
from gpiozero.pins.pigpio import PiGPIOFactory 
 
# Define the variable you want to check 
face_valid_person = False  # Change this to True or False to test the color 
change 
 
# Create the main window 
root = tk.Tk() 
root.title("Face Detection Status") 
 
 
def findFace(image,detector): 
    #img = cv.imread(image, cv.IMREAD_COLOR) 
    img = image 
    img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB) 
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) 
    faces_found = [] 
    faces = detector.detectMultiScale(gray,1.2,4,minSize=(100,100)) #size 
of image and likelyhood of image 
    for (x,y,w,h) in faces: 
      cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) 
      roi_gray = gray[y:y+h, x:x+h] 
      roi_colour = img_rgb[y:y+h, x:x+h] 
      faces_found.append(cv.resize(roi_colour, dsize=(126, 126, 3)[:2], 
interpolation=cv.INTER_AREA)) #return collection of faces found in the img 
    return faces_found 
 
 
 
def show_face_detected(face): 
    face_show = cv.cvtColor(face,cv.COLOR_BGR2RGB) 
    return face_show 
 
 
 
def flap(servoL, servoR): 
    print ('Wings up!') 
    # move wings up 
    for i in range(160,200): 
        servoL.value = math.sin(math.radians(i)) 
        servoR.value = math.sin(math.radians(i)-math.radians(180)) 
        sleep(0.05) 
    # hold position 
    sleep(0.5) 
    print ('Wings down!') 
    # move wings down 
    for i in range(200,160,-1): 
        servoL.value = math.sin(math.radians(i)) 
        servoR.value = math.sin(math.radians(i)-math.radians(180)) 
        sleep(0.05) 
    print ('Wings done!') 
 
 
 
 
haar_cascade = '/home/tom/FYP/haarcascade_frontalface_default.xml' 
hCascade = cv.CascadeClassifier(haar_cascade) 
 
# Servos 
factory = PiGPIOFactory() 
servoL = Servo(12, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000, 
pin_factory=factory) 
servoR = Servo(13, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000, 
pin_factory=factory) 
 
print('making interpreter') 
# Initialize the TF interpreter 
interpreter = 
edgetpu.make_interpreter('/home/tom/FYP/_final_model/converted_model.tflite
 ') 
print('allocating tensors') 
interpreter.allocate_tensors() 
 
print('starting .....') 
cap = cv.VideoCapture(0) 
common.set_input(interpreter,(126,126,3)) 
interpreter.invoke() 
 
skip_factor = 3 
frame_count = 0 
valid_face_chrono = 0 
while True: 
    #print('face loop') 
    ret, img = cap.read() 
    if not ret: 
        break 
    if frame_count % skip_factor == 0: 
        faces = findFace(img,hCascade) 
        if len(faces) >0: 
            test_face = faces[0] 
            common.set_input(interpreter, test_face) 
            interpreter.invoke() 
            classes = classify.get_scores(interpreter) 
            if classes[0] > 0.6: 
                print('face detected with confidence: 
',classes,str(time.strftime("(%a_%d_%b_%Y_%H:%M:%S)", time.gmtime()))) 
                valid_face_chrono += 1 
            else: 
                valid_face_chrono = 0 
        #print(faces) 
        #for f in range(0,len(faces)): 
        #    show_name = 'face' + str(f) 
        #    cv.imshow(show_name,show_face_detected(faces[f])) 
        cv.imshow('img', img) 
        face_valid_person = valid_face_chrono > 3 
    frame_count += 1 
    k = cv.waitKey(30) & 0xff
    if k == 27: 
        break 
#time.sleep(0.25) 
# Set the background color based on the variable's value 
    if face_valid_person: 
        root.configure(bg="green") 
    else: 
        root.configure(bg="red") 
    # Keep the window open until closed manually 
    root.update() 

#interpreter.release_tensors() 
interpreter.reset_all_variables()  # Optional for complete cleanup 
interpreter = None              
# Discard the interpreter object 
del interpreter 
root.destroy() 
cap.release() 
cv.destroyAllWindows() 
exit(1)
