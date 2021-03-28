import os
import cv2
import glob 
import moviepy
from pathlib import Path
import json

downloads_folder = '/Users/elsa/Downloads/'
data_folder = 'SisFall_Videos/'
adl_folder = 'ADLs/'
fall_folder = 'Falls/'
# Path to save the images
output_base_path = data_folder
# Label files, download them from the dataset's site
annotation_file = 'dataset_preprocessing/annotations_sisfall.json'
W, H = 224, 224 # shape of new images (resize is applied)
framespersecond = 29

#make output folders
if not os.path.exists(data_folder):
    os.makedirs(data_folder + fall_folder)
    os.makedirs(data_folder + adl_folder)

#input files
adl_files = glob.glob(downloads_folder + data_folder + adl_folder + '*')
fall_files = glob.glob(downloads_folder + data_folder + fall_folder + '*')

# processing ADL files 
# for video in adl_files: 
#     cap = cv2.VideoCapture(video)
#     length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     vidname = Path(video).stem + "/"
    
#     pos = 0
#     while pos < length:
#         ret, frame = cap.read()
#         start = 5 * framespersecond
        
#         pos += 1

#         output_path = ("./" + output_base_path + adl_folder + vidname)
#         if not os.path.exists(output_path):
#             os.makedirs(output_path)
#         if pos > start:
#             cv2.imwrite(output_path + 'img_{:05d}.jpg'.format(int(pos)),
#                 cv2.resize(frame, (W,H)),
#                 [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
#     cap.release()
#     cv2.destroyAllWindows()
with open(annotation_file, 'r') as json_file:
    annotations = json.load(json_file)

for video in fall_files: 
    cap = cv2.VideoCapture(video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vidname = Path(video).stem.lower()
    
    fall_starts = annotations[vidname]['start'] * framespersecond
    fall_ends = annotations[vidname]['end']  * framespersecond 

    pos = 0
    while pos < length:
        ret, frame = cap.read()
        start = 5 * framespersecond
        pos += 1

        if pos > start:
            if(pos < fall_starts or pos > fall_ends):
                output_path = ("./" + output_base_path + adl_folder + vidname + "/")
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                cv2.imwrite(output_path + 'img_{:05d}.jpg'.format(int(pos)),
                cv2.resize(frame, (W,H)),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                output_path = ("./" + output_base_path + fall_folder + vidname + "/")
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                cv2.imwrite(output_path + 'img_{:05d}.jpg'.format(int(pos)),
                cv2.resize(frame, (W,H)),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    cap.release()
    cv2.destroyAllWindows()

