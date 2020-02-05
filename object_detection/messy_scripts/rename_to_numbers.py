#!/usr/bin/env python
# coding: utf-8

# In[47]:


import os, glob


# In[73]:


get_ipython().system('ls')


# In[116]:


dir_path = "aug_1119_images_final_test/val_removed_labels_renamed"
# dir_path = "train_valid/valid"
# dir_path = "train_valid/train_real"
# dir_path = "train_valid/aug"
# dir_path = "train_valid/test"
# dir_path = "obj"

img_list = os.listdir(dir_path)
# img_list.pop(0)
# img_list.pop(0)
print(len(img_list))
print(img_list)


# In[119]:


img_list.remove("desktop.ini")
print(len(img_list))


# In[118]:


os.path.splitext(img_list[2])


# In[110]:


dir_path = "train_valid/aug_train_renamed"
for i in range(2020, 2020 + len(img_list), 2):
#     img_path = os.path.join(dir_path, img_list[i - 2020])
    txt_path = os.path.join(dir_path, img_list[i+1 - 2020])
#     os.rename(img_path, os.path.join(dir_path, '{:06}.jpg'.format(i//2)))
    os.rename(txt_path, os.path.join(dir_path, '{:06}.txt'.format(i//2)))


# In[120]:


dir_path = "aug_1119_images_final_test/val_removed_labels_renamed"
for i in range(1010, len(img_list)+1010):
#     img_path = os.path.join(dir_path, img_list[i - 2020])
    txt_path = os.path.join(dir_path, img_list[i - 1010])
#     os.rename(img_path, os.path.join(dir_path, '{:06}.jpg'.format(i//2)))
    os.rename(txt_path, os.path.join(dir_path, '{:06}.txt'.format(i)))


# In[ ]:




