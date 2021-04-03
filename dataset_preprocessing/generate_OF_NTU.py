import os
import cv2
import glob
import sys
from tqdm import tqdm
import numpy as np
import zipfile
import pandas as pd
import shutil
from generate_OF import generate_OF


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

folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
for folder in tqdm(folders):
    print("folder: " + folder)
    files = [f for f in os.listdir(data_folder + folder) if os.path.isfile(os.path.join(data_folder + folder, f))]
    curr_len = len(files)
    print(curr_len)
    if (curr_len < max_len):
        max_len = curr_len
    print(max_len)


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

