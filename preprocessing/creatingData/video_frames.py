# DOCUMENTATION
#   Purpose: Break video files into seperate frames
# Command Line Input:
#   python video_frames.py  --directory <directory path to store frames> --video <path to video file>
#   EXAMPLE:
#   python video_frames.py --directory ./underwater-video3-frames --video ./underwater_footage/video1.mp4

import cv2
import os
import argparse

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
    help="path to the video file")
ap.add_argument("-d", "--directory", required=True,
    help="path to the directory to store the frames. Will create a new directory \
    if needed")
args = vars(ap.parse_args())

# Capture each frame of the video file and save it to a directory
def FrameCapture(path, directory):
    vidObj = cv2.VideoCapture(path)
    vidObj_length = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    print("# of Frames: {}".format(vidObj_length))
    print("FPS: {}".format(vidObj.get(cv2.CAP_PROP_FPS)))

    padding = len(str(vidObj_length))
    count = 0

    while(vidObj.isOpened() and count < vidObj_length):
        success, image = vidObj.read()
        name = directory +'/{}.jpg'.format(str(count).rjust(padding,'0'))
        cv2.imwrite(name, image)
        print('Creating...{} -> {}'.format(name,success))
        count += 1

if __name__ == '__main__':

    try:
        if( not os.path.exists(args["directory"])):
            os.makedirs(args["directory"])
        FrameCapture(args["video"], args["directory"])
    except Exception as e:
        print("ERROR: ", str(e))
    else:
        print('Completed processing video {}'.format(args["video"]))
