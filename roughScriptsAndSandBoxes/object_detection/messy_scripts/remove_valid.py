#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os


# In[3]:


get_ipython().system('ls')


# In[73]:


dir_path = "yolo_labels/valid"
img_list = os.listdir(dir_path)
# img_list.pop(0)
# img_list.pop(0)
print(len(img_list))
img_list


# # To remove txt if there

# In[75]:


dir_path = "yolo_labels/train"
for name in img_list:
#     txt_name = os.path.splitext(name)[0] + ".txt"
    os.remove(os.path.join(dir_path, name))
img_list


# In[62]:


dir_path = "train_valid/valid_no_txt"
img_list = os.listdir(dir_path)
img_list.pop(0)
# img_list.pop(0)
print(len(img_list))
img_list


# In[63]:


for i in range(len(img_list)):
    img_list[i] = "aug_" + img_list[i]
#     img_name = os.path.splitext(img_list[i])
#     os.remove(os.path.join(aug_dir_path, img_list[i]))
img_list


# In[49]:


txt_list = []
for i in range(len(img_list)):
    img_name = os.path.splitext(img_list[i])[0] + '.txt'
    txt_list.append(img_name)
txt_list


# In[64]:


print(len(txt_list))
print(len(img_list))


# # Attempting to remove validation files

# In[77]:


aug_dir_path = "aug_1119_images_final_test/val_removed_images"
aug_dir_path_txt = "aug_1119_images_final_test/val_removed_labels"


# In[78]:


for i in range(len(img_list)):
    path = os.path.join(aug_dir_path, img_list[i])
    if os.path.isfile(path):
        os.remove(os.path.join(aug_dir_path, img_list[i]))


# In[79]:


for i in range(len(img_list)):
    path = os.path.join(aug_dir_path_txt, txt_list[i])
    if os.path.isfile(path):
        os.remove(os.path.join(aug_dir_path_txt, txt_list[i]))


# In[ ]:




