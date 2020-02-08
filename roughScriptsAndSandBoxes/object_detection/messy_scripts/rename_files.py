#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys


# In[17]:


directory = "1200_images_renamed_final"
sub_dir = "path_images"
sub_dir_path = os.path.join(directory, sub_dir)


# In[18]:


count = 0
for file_name in sorted(os.listdir(sub_dir_path), key=len):
    file_path = os.path.join(sub_dir_path, file_name)
#     print(file_path)
    os.rename(file_path, os.path.join(sub_dir_path, "path_{}.jpg".format(count)))
    count += 1


# In[ ]:




