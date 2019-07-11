# Example -> python labelling_frames.py --directory [path to frames folder]
# Prerequisites:
#   1) Frame numbering starts at 0 and is continuous until last frames
#       E.g 0,1,2,4,5,6 NOT 0,1,5,6,8

# Complete the list of labels
# Format -> [[start frame #, end frame #, label], [start frame #, end frame #, label]]
labels = [[5,6,'dog'],[7,7,'cat'], [8,500,'panda']]








import os
import argparse

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True,
    help="path to the directory that the video frames are stored in")
args = vars(ap.parse_args())

# Functions to rename the files with appropriate tags
def rename_frame_files(directory_path):
    files = sorted(os.listdir(directory_path))
    for i in range(0,len(labels)):
        for j in range(labels[i][0], labels[i][1]+1):
            fileName = files[j].split('.')
            # Remove old label if it exists
            if '_' in files[j]:
                temp = files[j].split('_')
                fileName = [temp[0]]
                temp = temp[1].split('.')
                fileName.append(temp[1])

            oldName = directory_path + files[j]

            newName = directory_path + fileName[0] +'_'+ labels[i][2] +'.'+ fileName[1]
            os.rename(oldName, newName)


# Executed this python moduleS
def main():
    rename_frame_files(args["directory"])
    # Will have two different formats for reading labels. One for humans, one for scripts

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        raise
    else:
        print('Completed labelling frames in directory: {}'.format(args["directory"]))
