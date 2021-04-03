#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import cv2
import glob
import sys
from tqdm import tqdm
import numpy as np
import zipfile
import pandas as pd
import shutil


# In[6]:


data_folder = 'C:/Users/Jordan Rahul/Desktop/NTU_FALL/'
dst_folder = 'C:/Users/Jordan Rahul/Desktop/NTU_FALL/dst_folder1/' # folder where the dataset is going to be unzipped
output_base_path = 'FDD_images/'
files = ['nturgbd_rgb_s001.zip']
W, H = 224, 224 # shape of new images (resize is applied)

if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# Extract zip files (if necessary)
for f in files:
    filepath = data_folder + f
    zfile = zipfile.ZipFile(filepath)
    pos = filepath.rfind('/')
    if not os.path.exists(dst_folder + filepath[pos+1:-4]):
        zfile.extractall(dst_folder)


# In[7]:


data_folder = 'C:/Users/Jordan Rahul/Desktop/NTU_FALL/dst_folder1/'
output_path = 'ntu001_OF/'

if not os.path.exists(output_path):
    os.mkdir(output_path)

folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
substring = "043"
FDD_videos = 'C:/Users/Jordan Rahul/Desktop/NTU_FALL/dst_folder1/FDD/'
max_len = 10000

for folder in tqdm(folders):
    print("folder: " + folder)
    files = [f for f in os.listdir(data_folder + folder) if os.path.isfile(os.path.join(data_folder + folder, f))]
    for file in files:
        #print(file)
        if substring in file:
            if not os.path.exists(FDD_videos):
                os.makedirs(FDD_videos)
            input_p = data_folder + folder + "/" + file
            shutil.move(input_p, FDD_videos)


# In[13]:


folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
for folder in tqdm(folders):
    print("folder: " + folder)
    files = [f for f in os.listdir(data_folder + folder) if os.path.isfile(os.path.join(data_folder + folder, f))]
    curr_len = len(files)
    print(curr_len)
    if (curr_len < max_len):
        max_len = curr_len
    print(max_len)


# In[14]:


def generate_OF(output_path, path_to_video):
    counter = 1
    cap = cv2.VideoCapture(path_to_video)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    while(1):
        ret, frame2 = cap.read()
        if ret:
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

          
            
            flow_x = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
            flow_y = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
            flow_x = flow_x.astype('uint8')
            flow_y = flow_y.astype('uint8')
            
            flow_x = cv2.resize(src=flow_x, dsize=(224, 224))
            flow_y = cv2.resize(src=flow_y, dsize=(224, 224))

            
            ##cv2.imshow('flow_x_img{:05d}.jpg'.format(counter), horz)
            ##cv2.imshow('flow_y_img{:05d}.jpg'.format(counter), vert)
            
            # show x and y direction windows in video 
            ##cv2.imshow('horizontal'.format(counter), horz)
            ##cv2.imshow('vertical'.format(counter), vert)

            cv2.imwrite(output_path + "/" + 'flow_x_img{:05d}.jpg'.format(counter), flow_x)
            cv2.imwrite(output_path + "/" + 'flow_y_img{:05d}.jpg'.format(counter), flow_y)
            
            
            k = cv2.waitKey(30) & 0xff

            if k == 27:
                break
            # elif k == ord('s'):
                # cv2.imwrite(output_path + "/" + 'opticalhsv_{:05d}.jpg'.format(counter),rgb)
            prvs = next
            counter +=1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


# In[ ]:


folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
for folder in tqdm(folders):
    print("folder: " + folder)
    files = [f for f in os.listdir(data_folder + folder) if os.path.isfile(os.path.join(data_folder + folder, f))]
    count = 0
    for file in tqdm(files):
        name = file.replace('.avi', '')
        output_p = output_path + folder + "/" + name
        input_p = data_folder + folder + "/" + file
        if not os.path.exists(output_p):
            os.makedirs(output_p)
        print(input_p)
        generate_OF(output_p, input_p)
        count +=1
        if (count > max_len):
            break


# In[ ]:




