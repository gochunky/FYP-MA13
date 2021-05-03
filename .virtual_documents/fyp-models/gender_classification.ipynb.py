# get_ipython().run_line_magic("pip", " install opencv-python")
# get_ipython().run_line_magic("pip", " install matplotlib")
# get_ipython().run_line_magic("pip", " install keras")
# get_ipython().run_line_magic("pip", " install seaborn")
# get_ipython().run_line_magic("pip", " install sklearn")


import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import os
import cv2
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPooling2D , Flatten , Dropout 
import seaborn as sns
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from ipykernel import kernelapp as app


epochs = 300
batch_size = 64
img_size = 224
labels = ['female', 'male']


def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data, dtype=list)


train = get_data('preprocessing/train_data')
val = get_data('preprocessing/val_data')
train_pert = get_data('preprocessing/makeup')


def plot_dataset_ratio(l, count, data, femaleLabel, maleLabel):
    for i in data:
        if (i[1] == 0):
            l.append(femaleLabel)      
        else:
            l.append(maleLabel)
        count += 1


l = []
train_count = 0
val_count = 0
train_pert_count = 0

plot_dataset_ratio(l, train_count, train, "female_train", "male_train")
plot_dataset_ratio(l, train_pert_count, train_pert, "female_pert", "male_pert")
plot_dataset_ratio(l, val_count, val, "female_val", "male_val")

pd.value_counts(pd.Series(data=l)).plot.bar()


x_train = []
y_train = []
x_train_pert = []
y_train_pert = []
x_val = []
y_val = []

# Unpertubed Dataset
for feature, label in train:
    x_train.append(feature)
    y_train.append(label)
x_train = np.array(x_train) / 255
x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

# Perturbed Dataset
for feature, label in train_pert:
    x_train_pert.append(feature)
    y_train_pert.append(label)
x_train_pert = np.array(x_train_pert) / 255
x_train_pert.reshape(-1, img_size, img_size, 1)
y_train_pert = np.array(y_train_pert)

# Validation Dataset
for feature, label in val:
    x_val.append(feature)
    y_val.append(label)
x_val = np.array(x_val) / 255
x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)


def createModel():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(224,224,3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    opt = Adam(lr=0.000001)
    model.compile(optimizer = opt , 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics = ['accuracy'])
    
    return model


# model.summary()


model = createModel()
checkpoint = ModelCheckpoint('model_best_weights.h5', monitor='loss', verbose=1, # Saves checkpoints
                             save_best_only=True, mode='min', save_freq='epoch')
history = model.fit(x_train,y_train,epochs = epochs ,
                    batch_size=batch_size,
                    validation_data = (x_val, y_val), 
                    callbacks = [checkpoint])


model_pert = createModel()
checkpoint_pert = ModelCheckpoint('model_best_weights_pert.h5', monitor='loss', verbose=1, # Saves checkpoints
                             save_best_only=True, mode='min', save_freq='epoch')
history_pert = model.fit(x_train_pert,y_train_pert,epochs = epochs ,
                    batch_size=batch_size,
                    validation_data = (x_val, y_val), 
                    callbacks = [checkpoint_pert])


target_names = ['Female (Class 0)','Male (Class 1)']
test_datagen = ImageDataGenerator(rescale = 1./255)
validation_generator = test_datagen.flow_from_directory("preprocessing/val_data",
                                                            target_size=(224, 224),
                                                            batch_size=batch_size,
                                                            class_mode='categorical')

def plot_train_val_acc_loss(history, noTrain, noVal):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    
def gen_cm(model, num_of_val_samples):
    Y_pred = model.predict(validation_generator, 2500 // batch_size+1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    confusion_matrix_array = confusion_matrix(validation_generator.classes, y_pred)
    print(confusion_matrix_array)
    df_cm = pd.DataFrame(confusion_matrix_array, range(2), range(2))
    # plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 12}) # font size
    plt.show()
    return y_pred


plot_train_val_acc_loss(history, train_count, val_count)


y_pred = gen_cm(model, val_count)


print(classification_report(validation_generator.classes, y_pred, target_names=target_names, zero_division=1))


plot_train_val_acc_loss(history, train_pert_count, val_count)


y_pred_pert = gen_cm(model_pert, val_count)


print(classification_report(validation_generator.classes, y_pred_pert, target_names=target_names, zero_division=1))


def predictImage(filename):
    img = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    input_arr = input_arr.astype('float32') / 255.  # This is VERY important
    predictions = model.predict(input_arr)
    val = np.argmax(predictions, axis=-1)
    
    img1 = image.load_img(filename,target_size=(img_size,img_size))
    plt.imshow(img1)
    if val == 1:
        plt.xlabel("MALE",fontsize=30)
    elif val == 0:
        plt.xlabel("FEMALE",fontsize=30)


predictImage("random.jpeg")


predictImage("celeb.jpg")


predictImage("emma.jpg")


predictImage("clairo.jpeg")


def predictImagePert(filename):
    img = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    input_arr = input_arr.astype('float32') / 255.  # This is VERY important
    predictions = model_pert.predict(input_arr)
    val = np.argmax(predictions, axis=-1)
    
    img1 = image.load_img(filename,target_size=(img_size,img_size))
    plt.imshow(img1)
    if val == 1:
        plt.xlabel("MALE",fontsize=30)
    elif val == 0:
        plt.xlabel("FEMALE",fontsize=30)


predictImagePert("random.jpeg")


predictImagePert("celeb.jpg")


predictImagePert("emma.jpg")


predictImagePert("clairo.jpeg")
