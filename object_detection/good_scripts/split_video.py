import os
import cv2

# sometimes open cv will write the image in a flipped orientation (opposite of video)
# can use this function or others to reverse it image = cv2.rotate(image, cv2.ROTATE_180)
def split_video_into_frames(video_path, output_dir, start_frame = 0, end_frame = -1):
    if not os.path.isfile(video_path):
        raise Exception("Label directory: {} is not a file".format(video_path))
    elif not os.path.isdir(output_dir):
        raise Exception("Output directory: {} is not a directory".format(output_dir))
    elif start_frame < 0:
        raise Exception("Start frame {} must be >= 1".format(start_frame))

    vid = cv2.VideoCapture(video_path) 

    if not vid:
        raise Exception("Cannot read video at {}".format(video_path))
        
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if end_frame > total_frames:
        raise Exception("End frame is out of bounds {} last frame = {}".format(start_frame, total_frames))
    elif end_frame < start_frame:
        raise Exception("End frame: {} is less than start frame: {}".format(end_frame, start_frame))

    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    success,image = vid.read()
    count = start_frame
    while success:
        # If images are flipped comment out or not below
        image = cv2.rotate(image, cv2.ROTATE_180)

        cv2.imwrite(os.path.join(output_dir, "{}_{}.jpg".format(os.path.splitext(video_path)[0], count)), image)

        success,image = vid.read()
        print('Read frame {}/{}'.format(count, total_frames))
        if count == end_frame:
            success = False
        count += 1

    vid.release()
    cv2.destroyAllWindows()