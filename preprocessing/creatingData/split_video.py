# DOCUMENTATION
#   Purpose: Break video files into seperate frames and label it
#   Note: This combines both the video_frames.py and the label_frames.py scripts
# Command Line Input:
#   python video_frames.py  --directory <directory path to store frames> --video <path to video file> --file <path to timestamp file>
#   EXAMPLE:
#   python video_frames.py --directory ./underwater-video3-frames --video ./underwater_footage/video1.mp4 --file ./test.csv
# Optional Arguments:
#   --compress: optionally compress image after saving

import cv2
import os
import argparse

from PIL import Image

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
    help="path to the video file")
ap.add_argument("-d", "--directory", required=True,
    help="path to the directory to store the frames. Will create a new directory \
    if needed")
ap.add_argument("-f", "--file", required=True,
    help="path to the text file containing timestamps for objects in video")
ap.add_argument("-c", "--compress", action="store_true",
    help="compress images before saving")
args = vars(ap.parse_args())

def getFrame(time, framerate):
    h, m, s = time.split(':')
    seconds = int(h) * 3600 + int(m) * 60 + int(s)
    return round(seconds * framerate)

def compressImage(path):
    image = Image.open(path)
    image.save(path, optimize=True, quality=70)
    print('Compressing...{} --> {}'.format(path, os.path.getsize(path)))

# Capture each frame of the video file and save it to a directory
def FrameCapture(path, directory, file):
    vidObj = cv2.VideoCapture(path)
    vidObj_length = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    vidObj_framerate = vidObj.get(cv2.CAP_PROP_FPS)
    print("# of Frames: {}".format(vidObj_length))
    print("FPS: {}".format(vidObj_framerate))

    padding = len(str(vidObj_length))
    count = 0

    file = open(args["file"], "r+")
    labels = list()
    header = file.readline()
    for line in file:
        line = line.rstrip().split(',')
        start = getFrame(line[0], vidObj_framerate)
        end = getFrame(line[1], vidObj_framerate)
        labels.append([start, end, line[2]])
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
        FrameCapture(args["video"], args["directory"], args["file"])
    except Exception as e:
        print("ERROR: ", str(e))
    else:
        print('Completed processing video {}'.format(args["video"]))
