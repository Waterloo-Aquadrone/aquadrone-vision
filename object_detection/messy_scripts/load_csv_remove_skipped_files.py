#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os, sys
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


# In[4]:


get_ipython().system('ls')


# In[176]:


label_data = pd.read_csv('final_labels_skipped_104_total_1119.csv', index_col="External ID")
label_data


# # Removes skipped files:

# In[182]:


label_data_skipped = label_data.loc[label_data['Label'] == '{}']
# label_data_skipped = label_data[label_data.Label != "{}"]
# label_data_skipped.sort_values('External ID')
label_data_skipped


# In[183]:


label_data = label_data[label_data.Label != "{}"]
label_data.sort_values('External ID')
label_data


# In[184]:


print(label_data.dtypes)


# In[185]:


label_col = label_data['Label']
label_col


# In[186]:


label_dict = label_col.to_dict()
label_dict


# In[187]:


for key in label_dict:
    print(key)
    label_dict[key] = json.loads(label_dict[key])
    print(label_dict[key])


# In[220]:


# test = label_dict['gate_20.jpg']
# data = json.loads(test)
# data
label_dict


# In[188]:


for key in label_dict:
    print(key)
    for value in label_dict[key]['objects']:
        print(value)
        print(value['title'])
        print(value['bbox'])


# In[85]:


for value in label_dict['gate_117.jpg']['objects']:
        print(value)
        print(value['title'])
        print(value['bbox'])


# In[90]:


img = cv2.imread('1200_images_renamed_final/all_images_together/gate_117.jpg')
1138+695


# In[92]:


cv2.rectangle(img, (954, 301), (954+485, 301+529),(255,0,0), 3)
cv2.rectangle(img, (303, 316), (303+684, 316+553),(0,255,255), 3)
cv2.rectangle(img, (303, 318), (303+1125, 318+551),(0,0, 255), 3)
plt.imshow(img)


# In[91]:


cv2.destroyAllWindows()
plt.clf()


# In[29]:


def convert_to_yolo(img_path, x1, y1, width, height):
    image = cv2.imread(img_path)
    image_width = image.shape[1]
    image_height = image.shape[0]
    x_center = (x1+width/2)/image_width
    y_center = (y1+height/2)/image_height
    rel_width = width/image_width
    rel_height = height/image_height
    return x_center, y_center, rel_width, rel_height


# In[163]:


x,y,z,k = convert_to_yolo('1200_images_renamed_final/path_images/path_247.jpg', 344, 170, 470, 377)

print(x)
print(y)
print(z)
print(k)
print(convert_to_yolo('1200_images_renamed_final/path_images/path_247.jpg', 344, 170, 470, 377))


# In[18]:


def yolo_to_coord(x, y, width, height, img_path):
    image = cv2.imread(img_path)
    img_w = image.shape[1]
    img_h = image.shape[0]
    x1, y1 = int((x + width/2)*img_w), int((y + height/2)*img_h)
    x2, y2 = int((x - width/2)*img_w), int((y - height/2)*img_h)
    return x1, y1, x2, y2


# In[19]:


q,w,e,r = yolo_to_coord(x, y, z, k, '1200_images_renamed_final/path_images/path_247.jpg')

print(q)
print(w)
print(e)
print(r)


# In[20]:


cv2.rectangle(img, (q, w), (e, r), (255,0,0), 3)
plt.imshow(img)


# In[221]:


def class_to_index(class_name):
    if class_name == "small_gate":
        return 0
    elif class_name == "big_gate":
        return 1
    elif class_name == "total_gate":
        return 2
    elif class_name == "path":
        return 3


# In[24]:


get_ipython().system('ls')


# In[105]:


print(len(label_dict))


# In[222]:


dir_path = os.path.join('1200_images_renamed_final', 'all_images_together')
# text_file = open('')
plt.figure(figsize=(20,600))
print(len(label_dict))
print("Red = Small Gate")
print("Yellow = Big Gate")
print("Purple = Path")
print("Green = Marker")

for i, key in enumerate(label_dict):
#     print(key)
    plt.subplot(100,3,i+1)
    plt.grid(False)
    plt.axis('off')
    img_path = os.path.join(dir_path, key)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for value in label_dict[key]['objects']:
        
        x = int(value['bbox']['left'])
        y = int(value['bbox']['top'])
        width = int(value['bbox']['width'])
        height = int(value['bbox']['height'])
        class_name = value['title']
        
        x2 = x+width
        y2 = y+height
        
        draw_rect(class_name, img, x, y, x2, y2)
        
#         print(convert_to_yolo(img_path, x, y, width, height))
#         print(img_path)
#         print ('{} {} {} {} {}'.format(class_to_index(value['title']), convert_to_yolo(img_path, x, y, width, height)))
#         print(value)
#         print(value['title'])
#         print(class_to_index(value['title']))
#         print(value['bbox'])
#     print('test')
    plt.title(key)
    plt.imshow(img)
    if i > 10: 
        break

plt.show()


# In[156]:


def draw_rect(class_name, img, x1, y1, x2, y2):
    if class_name == "total_gate":
        cv2.rectangle(img, (x1, y1), (x2, y2),(128,0,128), 3)
    elif class_name == "small_gate":
        cv2.rectangle(img, (x1, y1), (x2, y2),(255,0,0), 3)
    elif class_name == "big_gate":
        cv2.rectangle(img, (x1, y1), (x2, y2),(255,255,0), 3)
    elif class_name == "path":
        cv2.rectangle(img, (x1, y1), (x2, y2),(0,255,0), 3)


# In[198]:


get_ipython().system('ls')


# # Remove skipped files from folder:

# In[191]:


label_data_skipped


# In[196]:


skipped_names = label_data_skipped.index.values.tolist()
skipped_names


# In[213]:


cut_images_path = "1119_images_final"
for name in skipped_names:
    path_name = os.path.join(cut_images_path, name)
    print("Removed {}".format(path_name))
    os.remove(path_name)


# In[224]:


yolo_path = 'yolo_labels/train'
dir_path = '1119_images_final'
for key in label_dict:
    key_name = os.path.splitext(key)[0] + ".txt"
    print(key_name)
    text_path = os.path.join(yolo_path, key_name)
    text_file = open(text_path, 'w')
    img_path = os.path.join(dir_path, key)
    for value in label_dict[key]['objects']:
        x = int(value['bbox']['left'])
        y = int(value['bbox']['top'])
        width = int(value['bbox']['width'])
        height = int(value['bbox']['height'])
        class_name = value['title']
        class_val = class_to_index(class_name)
        x_c, y_c, w, h = convert_to_yolo(img_path, x, y, width, height)
#         print("{} {} {} {} {}".format(class_val, x_c, y_c, w, h))
        text_file.write("{} {} {} {} {}\n".format(class_val, x_c, y_c, w, h))


# In[218]:


for key in label_dict:
    print(key)
    for value in label_dict[key]['objects']:
        print(value)
        print(value['title'])
        print(value['bbox'])


# In[ ]:




