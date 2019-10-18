# DOCUMENTATION
#   Purpose: Break video files into seperate frames and label it
#   Note: This combines both the video_frames.py and the label_frames.py scripts
# Command Line Input:
#   python video_frames.py  --directory <directory path to store frames> --video <path to video file> 
#   EXAMPLE:
#   python video_frames.py --directory ./underwater-video3-frames --video ./underwater_footage/video1.mp4
# Optional Arguments:
#   --compress: optionally compress image after saving

import cv2
import os
import argparse
from PIL import Image

# Modify this to accustom to approximate fram numbers (will change to text file shortly)
labels = [[60,1423,'gate'],[1446,2640,'board'], [2704,4040,'dice'], [4045, 4725, 'board'],
          [4808, 5024, 'robot']]

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
    help="path to the video file")
ap.add_argument("-d", "--directory", required=True,
    help="path to the directory to store the frames. Will create a new directory \
    if needed")
ap.add_argument("-c", "--compress", action="store_true",
    help="compress images before saving")
args = vars(ap.parse_args())

def compressImage(path):
    image = Image.open(path)
    image.save(path, optimize=True, quality=40)
    print('Compressing...{} -> {}'.format(path, os.path.getsize(path)))

# Capture each frame of the video file and save it to a directory
def FrameCapture(path, directory):
    vidObj = cv2.VideoCapture(path)
    vidObj_length = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    print("# of Frames: {}".format(vidObj_length))
    print("FPS: {}".format(vidObj.get(cv2.CAP_PROP_FPS)))

    padding = len(str(vidObj_length))
    count = 0
    labels.sort()

    while(vidObj.isOpened() and count < vidObj_length):
        success, image = vidObj.read()

        # Filter array of labels to get correct frame label
        label = list(filter(lambda x: count >= x[0] and count <= x[1], labels))

        if (len(label) > 0 and success):
            subdirectory = directory + '/{}'.format(label[0][2])
            if not os.path.exists(subdirectory): os.makedirs(subdirectory)
            name = subdirectory + '/{}.jpg'.format(str(count).rjust(padding,'0'))
            cv2.imwrite(name, image)
            print('Creating...{} -> {}'.format(name,success))
            if args["compress"]: compressImage(name)
        count += 1

if __name__ == '__main__':

    try:
        if (not os.path.exists(args["directory"])):
            os.makedirs(args["directory"])
        FrameCapture(args["video"], args["directory"])
    except Exception as e:
        print("ERROR: ", str(e))
    else:
        print('Completed processing video {}'.format(args["video"]))
