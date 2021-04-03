import os
import glob 
from moviepy.editor import *
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
for video in adl_files: 
    # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    vidname = Path(video).stem + ".mp4"
    
    output_path = ("./" + output_base_path + adl_folder + vidname)
    clip = VideoFileClip(video)
    duration = int(clip.duration)
    clip = clip.subclip(5, duration)
    clip.write_videofile(output_path)
   

with open(annotation_file, 'r') as json_file:
    annotations = json.load(json_file)

for video in fall_files: 
    vidname_s = Path(video).stem.lower()

    vidname = Path(video).stem + ".mp4"
    
    fall_starts = annotations[vidname_s]['start']
    fall_ends = annotations[vidname_s]['end']  
    
    output_path = ("./" + output_base_path + fall_folder + vidname)
    clip = VideoFileClip(video)
    clip = clip.subclip(fall_starts, fall_ends)
    clip.write_videofile(output_path)
    