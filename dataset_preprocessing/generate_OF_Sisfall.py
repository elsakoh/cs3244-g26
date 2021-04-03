import os
import cv2
import glob
import sys
from tqdm import tqdm
from generate_OF import generate_OF

data_folder = 'Sisfall_Videos/'
output_path = 'SisFall_OF/'

if not os.path.exists(output_path):
    os.mkdir(output_path)

folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
for folder in tqdm(folders):
    print("folder: " + folder)
    files = [f for f in os.listdir(data_folder + folder) if os.path.isfile(os.path.join(data_folder + folder, f))]
    for file in files:
        name = file.replace('.mp4', '')
        output_p = output_path + folder + "/" + name
        input_p = data_folder + folder + "/" + file
        if not os.path.exists(output_p):
            os.makedirs(output_p)
        print(input_p)
        generate_OF(output_p, input_p)
        
