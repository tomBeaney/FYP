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


####################################################
###########       LOADING DATA          ############
####################################################

haar_cascade = '/content/haarcascade_frontalface_default.xml' #change as needed for the pi

test_img = '/content/drive/MyDrive/images/Almeida_Baptista_0001.jpg'
#test_img = '/content/download.png'
#cv.imshow('image',test_img)
'''
with zipfile.ZipFile('/content/images.zip','r') as zip_ref:
  zip_ref.extractall('/content')
'''
image_path = '/content/drive/MyDrive/face_images/faces_subset.pkl'
images_names = '/content/drive/MyDrive/face_images/names_subset.pkl'
#image_path = '/content/drive/MyDrive/face_images/faces.pkl'
#images_names = '/content/drive/MyDrive/face_images/names.pkl'
with open(image_path,'rb') as f:
  image_arrays = pickle.load(f)
with open(images_names,'rb') as f:
  image_names = pickle.load(f)
#print(images_names)
#test_names = fnmatch.filter(images_names,'A*.jpg')
#print(len(test_names),test_names)

#####################################################
############# MAKING TEST AND TRAIN SET #############
#####################################################

#X_data = images_names
X_data = image_arrays
Y_data = [0]*len(X_data)
#print(len(X_data),X_data)
#print(len(Y_data),Y_data)
subsetNames = ['Gerhard_Schroeder','Ariel_Sharon','Hugo_Chavez','Junichiro_Koizumi','Serena_Williams','Gloria_Macapagal_Arroyo','Arnold_Schwarzenegger',
                 'Jennifer_Capriati','Laura_Bush','Alejandro_Toledo','Kofi_Annan','Roh_Moo-hyun','Mahmoud_Abbas','Winona_Ryder','Saddam_Hussein',
                 'Tiger_Woods','Naomi_Watts','Jennifer_Aniston','Jiang_Zemin','Julianne_Moore','Venus_Williams','Hu_Jintao','Halle_Berry','Julie_Gerberding',
                 'Salma_Hayek','Wen_Jiabao','Charles_Moose','Yoriko_Kawaguchi','Kim_Ryong-sung','Nicanor_Duarte_Frutos','Jackie_Chan','Tang_Jiaxuan',
                 'Jiri_Novak','Sergio_Vieira_De_Mello','Muhammad_Ali','Mohammad_Khatami','Jesse_Jackson','Jeong_Se-hyun','Hugh_Grant','Hosni_Mubarak','Heizo_Takenaka']
def nameInApproved(test_name):
  try:
    return approvedNames.index(test_name[0:-9])
  except:
    return None

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

def custom_activation(x):
  """Custom activation function that maps values bigger than 0.5 to 1.0 and less than 0.5 to 0.0."""
  return tf.where(x >= 0.41, 1.0, x*0.001)




for trainingName in subsetNames:
#if True:
  approvedNames = [trainingName]
  #approvedNames = ['Hugo_Chavez']
  Y_approved = 0
  print(len(X_data))
  for index in range(0, len(X_data)):
    #print(image_names[index])
    if nameInApproved(image_names[index]) is not None:
      Y_data[index] = 1
      Y_approved += 1
      '''
      for j in range(0,1):
        X_data.append(X_data[index])
        Y_data.append(1)
        Y_approved += 1
      '''
    else:
      Y_data[index] = 0
  X_data_face = []
  Y_data_face = []
  X_data_face = X_data.copy()
  Y_data_face = Y_data.copy()
  loop_index = 0
  print(Y_approved)
  while Y_approved < 81:
    if loop_index <= len(X_data)-1:
      if Y_data[loop_index] == 1:
        X_data_face.append(X_data[loop_index])
        Y_data_face.append(Y_data[loop_index])
        Y_approved += 1
      loop_index += 1
    else:
      loop_index = 0

#########################################################
########## SPLITING DATASET INTO TRAIN AND TEST #########
#########################################################
  x_train, x_test, y_train, y_test = train_test_split(X_data_face, Y_data_face, test_size=0.10,
  random_state=54321)

  #print('training set: ',x_train)
  print('total number of faces: ',len(X_data_face))
  print('train set size:',len(x_train),'test set size: ',len(x_test))
  print('number of allowed faces: ',Y_approved,' number of non approved: ',(len(x_train)+len(x_test)-Y_approved))
  #print(Y_data)
  hCascade = cv.CascadeClassifier(haar_cascade)

  ##########################################
  ##########   MAKING MODEL  	##############
  ##########################################
  CNN_model= tf.keras.models.Sequential([
      tf.keras.Input(shape=(126,126,3)),
      tf.keras.layers.Rescaling(1./255), #normalize the image
      tf.keras.layers.Conv2D(8, (3,3), activation='relu'), #halfed from 16,32,64
      tf.keras.layers.MaxPool2D(2,2,padding='valid'),
      tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
      tf.keras.layers.MaxPool2D(2,2,padding='valid'),
      tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
      tf.keras.layers.MaxPool2D(2,2,padding='valid'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'), #was 512
      tf.keras.layers.Dense(1, activation='sigmoid')
      #tf.keras.layers.Activation(custom_activation)
  ],name=approvedNames[0])
  CNN_model.summary()
  x_train_r = []
  x_test_r = []
  y_train_r = []
  y_test_r = []
  print('loaded images \n\r starting training now .....')
  x_train_r=tf.convert_to_tensor(x_train,dtype=tf.float32)
  x_test_r=tf.convert_to_tensor(x_test,dtype=tf.float32)
  y_train_r=tf.convert_to_tensor(y_train,dtype=tf.float32)
  y_test_r=tf.convert_to_tensor(y_test,dtype=tf.float32)
  optimizer_CNN = tf.keras.optimizers.Adam(learning_rate=0.001)
  CNN_model.compile(optimizer=optimizer_CNN,loss=tf.losses.BinaryCrossentropy(),metrics=["binary_accuracy",
                                                                                          tf.keras.metrics.TruePositives(thresholds=0.9, name='tp'),
                                                                                          tf.keras.metrics.FalsePositives(thresholds=0.05, name='fp'),
                                                                                          tf.keras.metrics.TrueNegatives(thresholds=0.05, name='tn'),
                                                                                          tf.keras.metrics.FalseNegatives(thresholds=0.9, name='fn')])
  model_history = CNN_model.fit(x_train_r,y_train_r, batch_size=32,epochs=20,validation_data=(x_test_r,y_test_r),
                                verbose=1, shuffle=True)
  #if model is too complex then precision, recall and f1 score stop to calculate and just use the first values
  print('george bush classification: ',CNN_model.predict(tf.convert_to_tensor(
      [findFace('/content/drive/MyDrive/images/George_W_Bush_0001.jpg',hCascade)])))
  print('Ariel_Sharon classification: ',CNN_model.predict(tf.convert_to_tensor(
      [findFace('/content/drive/MyDrive/images/Ariel_Sharon_0001.jpg',hCascade)])))
  print('Arnold classification: ',CNN_model.predict(tf.convert_to_tensor(
      [findFace('/content/drive/MyDrive/images/Arnold_Schwarzenegger_0001.jpg',hCascade)])))
  test_face_path = '/content/drive/MyDrive/images/'+ approvedNames[0]+'_0001.jpg'
  test_face_result = findFace(test_face_path,hCascade)
  if test_face_result is not None:
    print('Test classification: ',CNN_model.predict(tf.convert_to_tensor(
        [test_face_result])))
  #############################
  ##    Saving model        ###
  #save_path_model = '/content/drive/MyDrive/models/model' + str(time.strftime("(%a_%d_%b_%Y_%H:%M:%S)", time.gmtime())) +'.tf'
  save_path_model = '/content/drive/MyDrive/models/model_' + approvedNames[0] + '.keras'
  #print()
  CNN_model.save(save_path_model)
  #needed
  '''
  converter = tf.lite.TFLiteConverter.from_saved_model(save_path_model)
  tflite_model = converter.convert()
  tflite_model_path = save_path_model + '/converted_model.tflite'
  with open(tflite_model_path,'wb') as f:
    f.write(tflite_model)
  '''
  #dont delete
  '''
  fig, ax = plt.subplots(ncols=4,figsize=(20,5))
  ax[0].plot(model_history.history['binary_accuracy'],color='teal',label='accuracy')
  ax[0].plot(model_history.history['val_binary_accuracy'],color='orange',label='val accuracy')
  ax[0].title.set_text('Accuracy')
  ax[0].legend()

  ax[1].plot(model_history.history['loss'],color='teal',label='loss')
  ax[1].plot(model_history.history['val_loss'],color='orange',label='val loss')
  ax[1].title.set_text('Loss')
  ax[1].legend()

  ax[2].plot(model_history.history['precision'],color='teal',label='precision')
  ax[2].plot(model_history.history['val_precision'],color='orange',label='val precision')
  ax[2].title.set_text('Precision')
  ax[2].legend()

  ax[3].plot(model_history.history['recall'],color='teal',label='recall')
  ax[3].plot(model_history.history['val_recall'],color='orange',label='val recall')
  ax[3].title.set_text('Recall')
  ax[3].legend()
  '''
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
  # plot the roc curve for the model
  plt.plot(fpr, tpr, linestyle='--', label=approvedNames[0])
  #plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
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
    """
    # Plot TPR vs. FPR.
    plt.plot(fpr, tpr, label='TPR vs. FPR')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.show()
    """
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
  '''
  precision=model_history.history['precision']
  val_precision= model_history.history['val_precision']
  recall= model_history.history['recall']
  val_recall= model_history.history['val_recall']
  f1_score= model_history.history['f1_score']
  val_f1_score= model_history.history['val_f1_score']
  '''
  epochs_range = range(20)

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
'''
plt.subplot(2, 2, 2)
plt.plot(epochs_range, tp, label='True Positive')
plt.plot(epochs_range, val_tp, label='Validation True Positive')
plt.legend(loc='upper right')
plt.title('Training and Validation True Positive')
plt.show()
plt.subplot(2, 3, 2)
plt.plot(epochs_range, fp, label='False Positive')
plt.plot(epochs_range, val_fp, label='Validation False Positive')
plt.legend(loc='upper right')
plt.title('Training and Validation False Positive')
plt.show()
'''
#plt.subplot(1, 3, 3)
#plt.plot(epochs_range, sensitivity_at_specificity, label='Training SAS')
#plt.plot(epochs_range, val_sensitivity_at_specificity, label='Validation SAS')
#plt.legend(loc='upper right')
#plt.title('ROC curve')
#plt.show()
'''
plt.subplot(2, 2, 1)
plt.plot(epochs_range, precision, label='Training precision')
plt.plot(epochs_range, val_precision, label='Validation precision')
plt.legend(loc='upper right')
plt.title('Training and Validation Precision')
plt.show()

plt.subplot(2, 2, 2)
plt.plot(epochs_range, recall, label='Training Recall')
plt.plot(epochs_range, val_recall, label='Validation Recall')
plt.legend(loc='upper right')
plt.title('Training and Validation Recall')
plt.show()

plt.subplot(3, 2, 1)
plt.plot(epochs_range, f1_score, label='Training f1_score')
plt.plot(epochs_range, val_f1_score, label='Validation f1_score')
plt.legend(loc='upper right')
plt.title('Training and Validation f1_score')
plt.show()
'''


'''
  k = cv.waitKey(30) &0xff
  if k == 27:
    break
img.release()
cv.destroyAllWindows()
#print(results)
'''
'''
def detect_face(image, scaleFactor, minNeighbors, minSize):
    # face will detected in gray image
    image_gray=cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    faces=fd.detect(image_gray,
                   scaleFactor=scaleFactor,
                   minNeighbors=minNeighbors,
                   minSize=minSize)

    for x, y, w, h in faces:
        #detected faces shown in color image
        cv.rectangle(image,(x,y),(x+w, y+h),(127, 255,0),3)

    show_image(image)
'''
