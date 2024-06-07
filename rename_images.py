# !/usr/bin/python
# coding: utf-8

"""A script to rename files with id_yy_mm_dd_HHMMSS_pose.png
Provide images in the following folder structure:

--images
  --id1
      --left
      --right
      --other
  --id2
      --left
      ... 

Id should be id-code or 'unmarked'

Usage: python rename_images.py images
"""

import os
import sys
import glob
import shutil
import datetime
from PIL import Image
from PIL.ExifTags import TAGS

input_dir = sys.argv[1]
input_dir = input_dir.replace("\\", "")
if input_dir[-1] == "/":
    input_dir = input_dir[:-1]

filetypes = [".JPG", ".jpg", ".PNG", ".png", ".JPEG", ".jpeg"]
poses = ["left", "right", "other"]
rename_files = list()

# list all image files from input directory
for root, dirs, files in os.walk(input_dir):
    for name in files:
        if name[-4:] in filetypes:
            path = os.path.join(root, name)
            rename_files.append(path)
        else:
            continue

# rename files inplace
count_rename = 0
for path in rename_files:
    dirname = os.path.dirname(path)
    dirname2 = os.path.dirname(dirname)
    pose = os.path.basename(dirname)
    Id = os.path.basename(dirname2)
    if pose in poses:
        image = Image.open(path)
        exifdata = image.getexif()
        created = exifdata.get(306)  # exif tag for image datetime
        created = datetime.datetime.strptime(str(created), "%Y:%m:%d %H:%M:%S")
        created = created.strftime("%y_%m_%d_%H%M%S")

        new_filename = "{}_{}_{}.png".format(Id, created, pose)
        new_path = os.path.join(dirname, new_filename)
        # If both src and dst are files,
        # dst will be replaced silently by os.rename
        # check if file exists first
        if os.path.is_file(new_path) == True:
            new_path = new_path[:-4] + "_COPY.png"
        os.rename(path, new_path)

        count_rename += 1

print("Renamed {} files.".format(count_rename))
