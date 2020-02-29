#!/usr/bin/env python
# coding: utf-8

# In[54]:


import os, json
import pandas as pd


# # Train, Validate Split

# In[105]:


valid_labels_path = "train_valid/valid_labels"
for name in os.listdir(valid_labels_path):
    train_path = "train_valid/train_labels"
    os.remove(os.path.join(train_path, name))


# # Train CSV

# In[106]:


train_path = "train_valid/train_labels"
train_list = os.listdir(train_path)
print(len(train_list))
print(train_list)


# In[107]:


file_name_list = []
width_list = []
height_list = []
class_list = []
xmin_list = []
ymin_list = []
xmax_list = []
ymax_list = []

for name in train_list:
#     print(key)
    file_path = os.path.join(train_path, name)
    file = open(file_path, 'r')
    file_data = json.load(file)
#     print(file_data)
    img_name = os.path.splitext(name)[0]
#     print(img_name)
    file_name_list.append(img_name)
    for key in file_data.values():
        if isinstance(key, dict):
#             print(key)
            width_list.append(key['width'])
            height_list.append(key['height'])
#         key_path = os.path.join(dir_path, key)
        elif len(key) != 0:
            if len(key) > 1:
                print("Greater than 1 {}".format(name))
                print(key)
            dict_obj = key[0]
            class_name = dict_obj['classTitle']
            class_list.append(class_name)
            
            box_dict = dict_obj['points']
            box_list = box_dict['exterior']
#             print(box_list)
#             if box_list == []:
# #                 print(key)
#                 print("name: {}".format(name))
            x1, y1 = box_list[0]
            x2, y2 = box_list[1]
            xmin_list.append(x1)
            ymin_list.append(y1)
            xmax_list.append(x2)
            ymax_list.append(y2)

print(len(file_name_list))
print(len(width_list))
print(len(height_list))
print(len(class_list))
print(len(xmin_list))
print(len(ymin_list))
print(len(xmax_list))
print(len(ymax_list))


# In[108]:


data_labels = {'filename': file_name_list, 'width': width_list, 'height': height_list, 
        'class': class_list, 'xmin': xmin_list, 'ymin': ymin_list, 'xmax':xmax_list, 'ymax':ymax_list}


# In[109]:


labels_df = pd.DataFrame(data = data_labels)
labels_df


# In[110]:


labels_df.isnull().values.any()


# In[62]:


get_ipython().system('ls')


# In[111]:


labels_df.to_csv("train_valid/train_labels.csv", index=None)


# # Valid CSV

# In[113]:


train_path = "train_valid/valid_labels"
train_list = os.listdir(train_path)
print(len(train_list))
print(train_list)


# In[114]:


file_name_list = []
width_list = []
height_list = []
class_list = []
xmin_list = []
ymin_list = []
xmax_list = []
ymax_list = []

for name in train_list:
#     print(key)
    file_path = os.path.join(train_path, name)
    file = open(file_path, 'r')
    file_data = json.load(file)
#     print(file_data)
    img_name = os.path.splitext(name)[0]
#     print(img_name)
    file_name_list.append(img_name)
    for key in file_data.values():
        if isinstance(key, dict):
#             print(key)
            width_list.append(key['width'])
            height_list.append(key['height'])
#         key_path = os.path.join(dir_path, key)
        elif len(key) != 0:
            if len(key) > 1:
                print("Greater than 1 {}".format(name))
                print(key)
            dict_obj = key[0]
            class_name = dict_obj['classTitle']
            class_list.append(class_name)
            
            box_dict = dict_obj['points']
            box_list = box_dict['exterior']
#             print(box_list)
#             if box_list == []:
# #                 print(key)
#                 print("name: {}".format(name))
            x1, y1 = box_list[0]
            x2, y2 = box_list[1]
            xmin_list.append(x1)
            ymin_list.append(y1)
            xmax_list.append(x2)
            ymax_list.append(y2)

print(len(file_name_list))
print(len(width_list))
print(len(height_list))
print(len(class_list))
print(len(xmin_list))
print(len(ymin_list))
print(len(xmax_list))
print(len(ymax_list))


# In[115]:


data_labels = {'filename': file_name_list, 'width': width_list, 'height': height_list, 
        'class': class_list, 'xmin': xmin_list, 'ymin': ymin_list, 'xmax':xmax_list, 'ymax':ymax_list}


# In[116]:


labels_df = pd.DataFrame(data = data_labels)
labels_df


# In[117]:


labels_df.isnull().values.any()


# In[118]:


labels_df.to_csv("train_valid/valid_labels.csv", index=None)


# In[ ]:




