########################
### MAKING SUPERMODEL ##
########################

from sys import maxsize
import tensorflow as tf
import keras
#from tensorflow import models,layers,utils
import cv2 as cv
from tensorflow.keras.models import *
from tensorflow.keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1, l2
import tqdm as tqdm
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow
import os
import zipfile
import numpy as np
#from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from tensorflow.keras.applications import VGG16
import fnmatch
import PIL
import pickle
from sklearn.metrics import roc_curve
from tensorflow.keras import Model, layers

####################################################
###########       LOADING DATA          ############
####################################################

haar_cascade = '/content/haarcascade_frontalface_default.xml' #change as needed for the pi

test_img = '/content/drive/MyDrive/images/Almeida_Baptista_0001.jpg'

image_path = '/content/drive/MyDrive/face_images/faces_subset.pkl'
image_name_path = '/content/drive/MyDrive/face_images/names_subset.pkl'
with open(image_path,'rb') as f:
  image_arrays = pickle.load(f)
with open(image_name_path,'rb') as f:
  image_names = pickle.load(f)

#####################################################
############# MAKING TEST AND TRAIN SET #############
#####################################################

#X_data = images_names
X_data = image_arrays
Y_data = [0]*len(X_data)
approvedNames = ['Naomi_Watts','Jennifer_Aniston','Jiang_Zemin','Julianne_Moore','Venus_Williams','Hu_Jintao','Halle_Berry','Julie_Gerberding',
                 'Salma_Hayek','Wen_Jiabao','Charles_Moose','Yoriko_Kawaguchi','Kim_Ryong-sung','Nicanor_Duarte_Frutos','Jackie_Chan','Tang_Jiaxuan',
                 'Jiri_Novak','Sergio_Vieira_De_Mello','Muhammad_Ali']
subsetNames = ['Gerhard_Schroeder','Ariel_Sharon','Hugo_Chavez','Junichiro_Koizumi','Serena_Williams','Gloria_Macapagal_Arroyo','Arnold_Schwarzenegger',
                 'Jennifer_Capriati','Laura_Bush','Alejandro_Toledo','Kofi_Annan','Roh_Moo-hyun','Mahmoud_Abbas','Winona_Ryder','Saddam_Hussein',
                 'Tiger_Woods','Naomi_Watts','Jennifer_Aniston','Jiang_Zemin','Julianne_Moore','Venus_Williams','Hu_Jintao','Halle_Berry','Julie_Gerberding',
                 'Salma_Hayek','Wen_Jiabao','Charles_Moose','Yoriko_Kawaguchi','Kim_Ryong-sung','Nicanor_Duarte_Frutos','Jackie_Chan','Tang_Jiaxuan',
                 'Jiri_Novak','Sergio_Vieira_De_Mello','Muhammad_Ali','Mohammad_Khatami','Jesse_Jackson','Jeong_Se-hyun','Hugh_Grant','Hosni_Mubarak','Heizo_Takenaka']
#subsetNames = approvedNames

model_path = '/content/drive/MyDrive/models/model_'

def nameInApproved(test_name):
  try:
    return approvedNames.index(test_name[0:-9])
  except:
    return None

Y_approved = 0
for index in range(0, len(X_data)):
  #print(image_names[index])
  if nameInApproved(image_names[index]) is not None:
    Y_data[index] = 1
    Y_approved += 1
  else:
    Y_data[index] = 0


#########################################################
########## SPLITING DATASET INTO TRAIN AND TEST #########
#########################################################

x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.10,random_state=54321)

#print('training set: ',x_train)
print('total number of faces: ',len(X_data))
print('train set size:',len(x_train),'test set size: ',len(x_test))
print('number of allowed faces: ',Y_approved,' number of non approved: ',(len(x_train)+len(x_test)-Y_approved))
#print(Y_data)

hCascade = cv.CascadeClassifier(haar_cascade)
def findFace(image,detector):
  img = cv.imread(image, cv.IMREAD_COLOR)
  img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
  gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
  faceFound = False
  faces = detector.detectMultiScale(gray,1.2,4,minSize=(100,100)) #size of image and likelyhood of image
  for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+h]
    roi_colour = img_rgb[y:y+h, x:x+h]
    faceFound = True
  if faceFound:
    return cv.resize(roi_colour, dsize=(126,126,3)[:2], interpolation=cv.INTER_AREA)
  else:
    return None


##########################################
##########   MAKING MODEL  	##############
##########################################

num_models = len(subsetNames)
sub_models = []
#loading in base models

# '/content/drive/MyDrive/models/model_Ariel_Sharon/converted_model.tflite'
for file_name in subsetNames:
  model_file_path = model_path + file_name + '.keras'
  new_model = tf.keras.models.load_model(model_file_path)
  #print(new_model.input_shape)
  #new_model.name = file_name
  new_model.trainable= False
  sub_models.append(new_model)

#print(sub_models)
'''
CNN_model = Sequential([
    layers.Input(shape=(126,126,3)),
    sub_models[0],
    sub_models[1],
    layers.Dense(1,activation='sigmoid'),
])
'''
arnie_face = tf.convert_to_tensor(
    [findFace('/content/drive/MyDrive/images/George_W_Bush_0001.jpg',hCascade)])
max_output= [0,'none']
total_output = 0
for smodel in sub_models:
  smodel_pred = smodel.predict(arnie_face)
  print(smodel.name,' classification: ',smodel_pred)
  total_output += smodel_pred
  if smodel_pred >= max_output[0]:
    max_output = smodel_pred,smodel.name

print(max_output,' the total output value is: ',total_output)


epoch_number = 40
inputs = layers.Input(shape=((126,126,3)))
#x = layers.Concatenate(-1)(inputs)  # Concatenate outputs of sub-models
# Extract outputs from sub-models (assuming sub_models are already defined)
#rescaling = layers.Rescaling(1./255)(inputs) #normalize the image
sub_model_outputs = []
for sub_model in sub_models:
  sub_model_outputs.append(sub_model(inputs))
# Concatenate sub-model outputs along axis=-1
x = layers.Concatenate(axis=-1)(sub_model_outputs)
#max_output = tf.math.reduce_max(x,axis=1)
def custom_relu_thresh_max(x, threshold=0.9, max_val=1.0):
  """Custom activation function with ReLU threshold and max value clipping."""
  # Apply ReLU
  relu = tf.nn.relu(x)
  # Clip to threshold value
  clipped_relu = tf.maximum(relu, threshold * relu)
  # Clip to max value
  return tf.clip_by_value(clipped_relu, clip_value_min=0.0, clip_value_max=max_val)


def relu_positive_clipping(x):
  relu = tf.nn.relu(x)
  return tf.where(relu >=1.0,1.0,relu)



#y = layers.LocallyConnected1D(32,3,activation=custom_relu_thresh_max,input_shape=(None,41))(x)
#dropout_Connections = layers.Dropout(0.3)(y)
#maybe have a piecewise function instead of relu
#flat = layers.Flatten()(x)
#filter_layer = layers.Activation(custom_relu_thresh_max)(x)
#dense_layer = layers.Dense(1,activation=relu_positive_clipping)(x)
#outputs = layers.Dropout(0.05)(dense_layer)
outputs = layers.Dense(1,activation=relu_positive_clipping)(x)
#outputs = layers.Lambda(lambda x: tf.reduce_max(x, axis=1),output_shape=(None,1))  # Apply max reduction along axis 1 (features)

# Create the final model
CNN_model = Model(inputs=inputs, outputs=outputs, name='Final_Model')

CNN_model.summary()

x_train_r=tf.convert_to_tensor(x_train,dtype=tf.float32)
x_test_r=tf.convert_to_tensor(x_test,dtype=tf.float32)
y_train_r=tf.convert_to_tensor(y_train,dtype=tf.float32)
y_test_r=tf.convert_to_tensor(y_test,dtype=tf.float32)

print(len(x_train_r),len(y_train_r),len(x_test_r),len(y_test_r))

optimizer_CNN = tf.keras.optimizers.Adam(learning_rate=0.003)
CNN_model.compile(optimizer=optimizer_CNN,loss=tf.losses.BinaryCrossentropy(),metrics=["binary_accuracy",
                                                                                        tf.keras.metrics.TruePositives(thresholds=0.9, name='tp'), #0.9
                                                                                        tf.keras.metrics.FalsePositives(thresholds=0.05, name='fp'), #0.05
                                                                                        tf.keras.metrics.TrueNegatives(thresholds=0.05, name='tn'), #0.05
                                                                                        tf.keras.metrics.FalseNegatives(thresholds=0.9, name='fn')]) #0.9

model_history = CNN_model.fit(x_train_r,y_train_r, batch_size=32,epochs=epoch_number,validation_data=(x_test_r,y_test_r),
                              verbose=1, shuffle=True)
#if model is too complex then precision, recall and f1 score stop to calculate and just use the first values

#print(max_output)


print('george bush classification: ',CNN_model.predict(tf.convert_to_tensor(
    [findFace('/content/drive/MyDrive/images/George_W_Bush_0001.jpg',hCascade)])))
print('tony blair classification: ',CNN_model.predict(tf.convert_to_tensor(
    [findFace('/content/drive/MyDrive/images/Tony_Blair_0001.jpg',hCascade)])))
print('Yoriko_Kawaguchi classification: ',CNN_model.predict(tf.convert_to_tensor(
    [findFace('/content/drive/MyDrive/images/Yoriko_Kawaguchi_0001.jpg',hCascade)])))
print('Arnold classification: ',CNN_model.predict(tf.convert_to_tensor(
    [findFace('/content/drive/MyDrive/images/Arnold_Schwarzenegger_0001.jpg',hCascade)])))
try:
  print('Jennifer Aniston classification: ',CNN_model.predict(tf.convert_to_tensor(
      [findFace('/content/drive/MyDrive/images/Jennifer_Aniston_0001.jpg',hCascade)])))
except:
  print('Could not find Jennifer_Aniston')
#print(CNN_model.metric_names)


save_path_model = '/content/drive/MyDrive/models/_final_model'
#print()
CNN_model.export(save_path_model)
converter = tf.lite.TFLiteConverter.from_saved_model(save_path_model)
tflite_model = converter.convert()
tflite_model_path = save_path_model + '/converted_model.tflite'
with open(tflite_model_path,'wb') as f:
  f.write(tflite_model)

#dont delete
acc = model_history.history['binary_accuracy']
val_acc = model_history.history['val_binary_accuracy']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

#val_sensitivity_at_specificity=model_history.history['val_sensitivity_at_specificity']
tp = model_history.history['tp']
#sensitivity_at_specificity= model_history.history['sensitivity_at_specificity']
val_tp = model_history.history['val_tp']
fp= model_history.history['fp']
val_fp= model_history.history['val_fp']
tn = model_history.history['tn']
fn = model_history.history['fn']
val_tn = model_history.history['val_tn']
val_fn = model_history.history['val_fn']

y_pred = CNN_model.predict(x_test_r)
fpr, tpr, _ = roc_curve(y_test_r,y_pred)
ns_probs = [0 for _ in range(len(y_test_r))]
ns_fpr, ns_tpr, thres = roc_curve(y_test_r, ns_probs)
# plot the roc curve for the model
plt.plot(fpr, tpr, linestyle='--', label='Trained Model')
plt.plot(ns_fpr, ns_tpr, marker='.', label='No Skill')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
print(_)



'''
def plot_tpr_fpr(true_positives, false_positives,true_negative,false_negative,name):
  """
  This function calculates the True Positive Rate (TPR) and False Positive Rate (FPR)
  for a given set of true positive and false positive data. It then plots the TPR vs. FPR.

  Args:
    true_positives: A list of true positive values.
    false_positives: A list of false positive values (can be of different size).
  """

  # Calculate TPR and FPR for each corresponding pair of true positives and false positives.
  tpr_fpr_data = []
  for tp, fp, tn, fn in zip(true_positives, false_positives,true_negative,false_negative):
    # Assuming total number of positives is the sum of true positives and false positives.
    total_positives = tp + fp
    #True Positive Rate = True Positives / (True Positives + False Negatives)
    tpr = tp/ (tp + fn)
    #False Positive Rate = False Positives / (False Positives + True Negatives)
    fpr = fp / (fp + tn)
    #tpr = tp / total_positives if total_positives > 0 else 0
    #fpr = fp / total_positives if total_positives > 0 else 0
    tpr_fpr_data.append((fpr, tpr))

  # Extract FPR and TPR values for plotting.
  fpr, tpr = zip(*tpr_fpr_data)
  return fpr, tpr
#lr_probs = model.predict_proba(testX)
#lr_fpr, lr_tpr, _ = roc_curve(y_test_r, lr_probs)
fpr, tpr = plot_tpr_fpr(tp,fp,tn,fn,'ROC Curve')
val_fpr, val_tpr = plot_tpr_fpr(val_tp,val_fp,val_tn,val_fn,'Validation ROC Curve')
fpr,tpr = zip(*sorted(zip(fpr,tpr)))

val_fpr,val_tpr = zip(*sorted(zip(val_fpr,val_tpr)))

plt.plot(fpr,tpr, marker='.', label='Training')
plt.plot(val_fpr, val_tpr, linestyle='--', label='Validation')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
'''
epochs_range = range(epoch_number)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
