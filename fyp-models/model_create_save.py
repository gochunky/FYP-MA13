#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from gen_results import gen_save_cr_cm # Load test results
import pandas as pd
import seaborn as sns

import json
import ssl
import time
ssl._create_default_https_context = ssl._create_unverified_context


# In[2]:


global preprocessing_fp
global train_dataset
global validation_dataset
global data_augmentation

global EPOCHS
global flag # Tracks whether base_models where initialised
flag = False

AUTOTUNE = tf.data.AUTOTUNE # Use buffered prefetching to load images without having I/O blocking
BATCH_SIZE = 32
EPOCHS = 50
IMG_SIZE = (224,224)
IMG_SHAPE = IMG_SIZE + (3,)
LABELS = ["female", "male"]

# Load base models
global preprocess_input_mobile, base_model_mobile
global preprocess_input_dense, base_model_dense
global preprocess_input_res, base_model_res

# Generate models
global model_mobile
global model_dense
global model_res


# In[3]:

def set_preprocessing_fp(target):
    global preprocessing_fp 
    preprocessing_fp = target
    
def set_epochs(epochs=50):
    global EPOCHS
    EPOCHS = epochs


# In[4]:


def load_train_val():
    global train_dataset
    global validation_dataset
    
    # Load train dataset
    train_dataset = image_dataset_from_directory(os.path.join(preprocessing_fp, "train"),
                                                 shuffle=True,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE)

    # Load validation dataset
    validation_dataset = image_dataset_from_directory(os.path.join(preprocessing_fp, "val"),
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE,
                                                      image_size=IMG_SIZE)


# In[5]:


def data_prefetch_augmentation():
    global train_dataset
    global validation_dataset
    global data_augmentation 
    
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    # Helpful since we want to expand our image dataset
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])


# In[6]:


def load_base_models(IMG_SHAPE=IMG_SHAPE):
    global preprocess_input_mobile, base_model_mobile
    global preprocess_input_dense, base_model_dense
    global preprocess_input_res, base_model_res
    
    # Load MobileNetV3 Large
    preprocess_input_mobile = tf.keras.applications.mobilenet_v3.preprocess_input
    base_model_mobile = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

    # Load DenseNet 201
    preprocess_input_dense = tf.keras.applications.densenet.preprocess_input
    base_model_dense = tf.keras.applications.densenet.DenseNet201(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

    # Load ResNet50
    preprocess_input_res = tf.keras.applications.resnet50.preprocess_input
    base_model_res = tf.keras.applications.resnet50.ResNet50(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


# # Model Training

# In[7]:


def create_model(base_model, preprocess_input):
    global train_dataset
    
    # Converts images into a 5x5x1280 block of features
    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    
    # Freeze all convolutional base
    base_model.trainable = False
    
    # Add classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    
    # Model building
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(1024, kernel_regularizer='l2', activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1,activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    # Compile the model
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


# In[8]:


def gen_models():
    global model_mobile
    global model_dense
    global model_res
    global base_model_mobile, preprocess_input_mobile
    global base_model_dense, preprocess_input_dense
    global base_model_res, preprocess_input_res
    
    # Create the three different models
    model_mobile = create_model(base_model_mobile, preprocess_input_mobile)
    model_dense = create_model(base_model_dense, preprocess_input_dense)
    model_res = create_model(base_model_res, preprocess_input_res)


# In[9]:


def model_training(model, set_no, suffix):
    """
    Trains model, and saves model's best weights and history
    
    set_no: int
        set number
    """
    checkpoint = ModelCheckpoint(
        "best_weights/set{}/model_tl_best_weights_{}_set{}.h5".format(set_no, suffix, set_no),
        monitor="loss",
        verbose=0,
        save_best_only=True,
        mode="min",
        save_freq="epoch",
    )
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=30)
    
    # Save a checkpoint of the model for later use
    start_time = time.time()
    history = model.fit(train_dataset,
                             epochs=EPOCHS,
                             validation_data=validation_dataset,
                            callbacks=[early_stopping, checkpoint])
    time_taken = "%.2fs" % (time.time() - start_time)
    history.history['time_taken'] = time_taken

    # Store model history as a JSON file
    target = "history/set{}".format(str(set_no))
    with open(os.path.join(target, "model_tl_history_{}_set{}.json".format(suffix, set_no)), "w") as f:
        json.dump(history.history, f) # Construct the baseline (unperturbed) model
        
    return history


# In[10]:


def reset(target, epochs=EPOCHS):
    """
    Initialises or resets datasets according to target path
    
    """
    global flag
    
    print("Setting preprocessing_fp...")
    set_preprocessing_fp(target)
    print("Setting number of epochs...")
    set_epochs(epochs)
    print("Loading train and validation data...")
    load_train_val()
    if not flag:
        print("Loading prefetch and data augmentation variable initialised...")
        data_prefetch_augmentation()
        print("Loading base models...")
        load_base_models()
        flag = True
        print("Flag set to:", flag)
    print("Generating models...")
    gen_models()


# In[ ]:


# Train each of the models
def find_best_weights_and_history(set_no):
    """
    Gets model best weights from training and history
    
    """
    history_mobile = model_training(model_mobile,  set_no, 'mobile')
    history_dense = model_training(model_dense, set_no, 'dense')
    history_res = model_training(model_res, set_no, 'res')


# In[ ]:





# In[ ]:




