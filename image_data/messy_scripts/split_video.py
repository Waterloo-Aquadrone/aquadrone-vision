#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import cv2


# In[2]:


# sys.path.append("assets/bad_data")
get_ipython().system('ls')


# In[7]:


sys.path


# In[3]:


start_frame = 0;
directory = "test_video"


# In[5]:


for video_name in os.listdir(directory):
    video_name_path = os.path.join(directory, video_name)
    if os.path.isfile(video_name_path):
        print(video_name_path)
        vid = cv2.VideoCapture(video_name_path) 
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    #   vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        success,image = vid.read()
        count = 0

        while success:
            # image = image[:,100:1920-100]
            # for some reason some of the videos are rotated when opened
    #         image = cv2.rotate(image, cv2.ROTATE_180)
            cv2.imwrite("test_video/video_1_images/{}_{}.jpg".format(video_name, count), image) # save frame as JPEG file   

            success,image = vid.read()
            print('Read frame {}/{}'.format(count, total_frames))
    #         if count > 10:
    #           success = False
            count += 1


# In[ ]:


# video_name = "output1_Trim.mp4"
# video_name_path = os.path.join(directory, video_name)
# print(video_name_path)
# vid = cv2.VideoCapture(video_name_path) 
# total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
# #   vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
# success,image = vid.read()
# count = 0

# while success:
#     # image = image[:,100:1920-100]
#     # for some reason some of the videos are rotated when opened
# #        image = cv2.rotate(image, cv2.ROTATE_180)
#     cv2.imwrite("gate_images/{}_{}.jpg".format(video_name, count), image) # save frame as JPEG file   

#     success,image = vid.read()
#     print('Read frame {}/{}'.format(count, total_frames))
# #       if count > 10:
# #       success = False
#     count += 1


# In[21]:


vid.release()
cv2.destroyAllWindows()


# In[ ]:




