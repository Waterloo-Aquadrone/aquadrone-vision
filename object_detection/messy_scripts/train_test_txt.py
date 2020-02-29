#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os


# In[4]:


get_ipython().system('ls')


# In[28]:


img_list = os.listdir("train_valid/train_real_renamed")

# num = int(os.path.splitext(img_list[x])[0].split('_', 1)[1])
# print(num)
# img_list_sorted = img_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
# img_list
img_list


# In[30]:


len(img_list)


# In[31]:


img_list.remove("desktop.ini")


# In[32]:


len(img_list)


# In[33]:


text_file = open("train.txt", "w")


# In[22]:


# for i in range(len(img_list)):
#     ext = os.path.splitext(img_list[i])[1]
#     if ext == ".txt":
#         img_list.remove(img_list[i])
# img_list


# In[34]:


print(img_list)


# In[35]:


len(img_list)


# In[36]:


for img in img_list:
    ext = os.path.splitext(img)[1]
    if ext == ".jpg":
        img_path = os.path.join("data/obj", img) + "\n"
        text_file.write(img_path)


# In[ ]:




