#!/usr/bin/env python
# coding: utf-8
# In[1]:


import pandas as pd
import dlib
from tqdm import tqdm
import os
import itertools
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import face_recognition
# from perturb_filters.glasses.put_glasses import apply_glasses


# In[2]:


LABELS = ['female', 'male']


# In[28]:




def apply_makeup(parent_folder, image_fn ,output_fn):
    """
    Applies make up filter on a single image and saves it to a given directory.
    
    parent_folder : str
        Parent folder of image
    image_fn : str
        Filename of image
    output_fn : str
        Output image to save
    """
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(parent_folder+"/"+image_fn)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    pil_image = Image.fromarray(image)
    for face_landmarks in face_landmarks_list:
        d = ImageDraw.Draw(pil_image, 'RGBA')

        # Make the eyebrows into a nightmare
        d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
        d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
        d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
        d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

        # Gloss the lips
        d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
        d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
        d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
        d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

        # Sparkle the eyes
        d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
        d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

        # Apply some eyeliner
        d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
        d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)
        fn = Path(output_fn + image_fn)
        if not fn.is_dir():
            pil_image.save(output_fn + "/" + image_fn)    # Change this to male or female


# In[29]:


def gen_test_makeup():
    """
    Generates makeup test datasets for each CV set
    
    """
    original = 'cv_datasets/'
    folder = os.listdir(original)
    for set_fn in folder:
        for gender in LABELS:
            ori_path = original + set_fn + '/test/' + gender + '/'
            temp = os.listdir(ori_path)
            for i in tqdm(range(len(temp)), "Generating makeup for {} images...".format(gender)):
                image_fn = temp[i]
                target = original + set_fn + "/test_"
                target_makeup = target+'makeup/'+ gender + '/'

                # Generate make up dataset
                    # Create new folder if doesn't already exist
                Path(target_makeup).mkdir(parents=True, exist_ok=True)    
                # Save image in this folder
                    # If doesn't already exist
                if not Path(target_makeup + image_fn).is_dir():
                    apply_makeup(ori_path, image_fn, target_makeup)

# In[ ]:




