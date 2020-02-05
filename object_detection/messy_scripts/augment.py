#!/usr/bin/env python
# coding: utf-8

# ## Tutorial : https://medium.com/@a.karazhay/guide-augment-images-and-multiple-bounding-boxes-for-deep-learning-in-4-steps-with-the-notebook-9b263e414dac

# In[1]:


import imgaug as ia
ia.seed(1)
# imgaug uses matplotlib backend for displaying images
get_ipython().run_line_magic('matplotlib', 'inline')
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa 
# imageio library will be used for image input/output
import imageio
import pandas as pd
import numpy as np
import re
import os
import glob
# this library is needed to read XML files for converting it into CSV
# import xml.etree.ElementTree as ET
import shutil
import json


# In[4]:


label_data = pd.read_csv('final_labels_skipped_104_total_1119.csv', index_col="External ID")
# label_data


# In[7]:


label_data = label_data[label_data.Label != "{}"]
label_data.sort_values('External ID')
# label_data


# In[8]:


label_col = label_data['Label']
# label_col


# In[9]:


label_dict = label_col.to_dict()
# label_dict


# In[10]:


for key in label_dict:
#     print(key)
    label_dict[key] = json.loads(label_dict[key])
#     print(label_dict[key])


# In[26]:


# for i, file in enumerate(os.listdir(dir_path)):
#     file_path = os.path.join(dir_path, file)
#     print(imageio.imread(file_path).shape[0])
#     if i > 30:
#         break


# In[36]:


dir_path = "1119_images_final_test"

file_name_list = []
width_list = []
height_list = []
class_list = []
xmin_list = []
ymin_list = []
xmax_list = []
ymax_list = []
count = 1
for key in label_dict:
#     print(key)
    count +=1
    for value in label_dict[key]['objects']:
        key_path = os.path.join(dir_path, key)
        
        file_name_list.append(key)

        class_name = value['title']
        
        width = imageio.imread(key_path).shape[0]
        height = imageio.imread(key_path).shape[1]

        box_width = int(value['bbox']['width'])
        box_height = int(value['bbox']['height'])
        
        x1 = int(value['bbox']['left'])
        y1 = int(value['bbox']['top'])
        
        x2 = x1 + box_width
        y2 = y1 + box_height
        
        x1, x2 = max_pair(x1, x2)
        y1, y2 = max_pair(y1, y2)
        
#       print('{} {} {} {} {} {} {} {}'.format(key, width, height, class_name, x1, y1, x2, y2))
        width_list.append(width)
        height_list.append(height)
        class_list.append(class_name)
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
print(count)


# In[12]:


def max_pair(x1, x2):
    if x1 < x2:
        return x1, x2
    elif x1 > x2:
        return x2, x1
    
    return x1, x2


# In[37]:


data_img = {'filename': file_name_list, 'width': width_list, 'height': height_list, 
        'class': class_list, 'xmin': xmin_list, 'ymin': ymin_list, 'xmax':xmax_list, 'ymax':ymax_list}


# In[38]:


labels_df = pd.DataFrame(data = data_img)
labels_df


# In[39]:


# apply xml_to_csv() function to convert all XML files in images/ folder into labels.csv
labels_df.to_csv(('labels.csv'), index=None)
print('Successfully converted xml to csv.')


# In[40]:


grouped = labels_df.groupby('filename')


# In[41]:


group_df = grouped.get_group('gate_174.jpg')
group_df = group_df.reset_index()
group_df = group_df.drop(['index'], axis=1)
group_df


# In[42]:


# get bounding boxes coordinates from grouped data frame and write into array        
bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
# display the array we've got
bb_array


# In[34]:


# load images as NumPy arrays and append them to images list
images = []
for index, file in enumerate(glob.glob('1119_images_final/*.jpg')):
    images.append(imageio.imread(file))
    print(imageio.imread(file).shape)
    if index > 130:
        break
    
# how many images we have
print('We have {} images'.format(len(images)))


# In[20]:


# what are the sizes of the images
for index, file in enumerate(glob.glob('1119_images_final_test/*.jpg')):
    print('Image {} have size of {}'.format(file[7:], images[index].shape))
    print(index)
    if index > 30:
        break


# In[43]:


# pass the array of bounding boxes coordinates to the imgaug library
bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=images[63].shape)
# display the image and draw bounding boxes
ia.imshow(bbs.draw_on_image(images[63], size=2))


# In[44]:


# to resize the images we create two augmenters
# one is used when the image height is more than 600px and the other when the width is more than 600px
height_resize = iaa.Sequential([ 
    iaa.Resize({"height": 600, "width": 'keep-aspect-ratio'})
])

# our data mainly has 1920x1080
# the others are 640x480
width_resize = iaa.Sequential([ 
    iaa.Resize({"height": 'keep-aspect-ratio', "width": 600})
])


# In[48]:


# function to convert BoundingBoxesOnImage object into DataFrame
def bbs_obj_to_df(bbs_object):
#     convert BoundingBoxesOnImage object into array
    bbs_array = bbs_object.to_xyxy_array()
#     convert array into a DataFrame ['xmin', 'ymin', 'xmax', 'ymax'] columns
    df_bbs = pd.DataFrame(bbs_array, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    return df_bbs


# In[50]:


def resize_imgaug(df, images_path, aug_images_path, image_prefix):
    # create data frame which we're going to populate with augmented image info
    aug_bbs_xy = pd.DataFrame(columns=
                              ['filename','width','height','class', 'xmin', 'ymin', 'xmax', 'ymax']
                             )
    grouped = df.groupby('filename')    
    
    for filename in df['filename'].unique():
    #   Get separate data frame grouped by file name
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)
        
    #   The only difference between if and elif statements below is the use of height_resize and width_resize augmentors
    #   deffined previously.

    #   If image height is greater than or equal to image width 
    #   AND greater than 600px perform resizing augmentation shrinking image height to 600px.
        if group_df['height'].unique()[0] >= group_df['width'].unique()[0] and group_df['height'].unique()[0] > 600:
        #   read the image
            image = imageio.imread(images_path+filename, pilmode="RGB")
        #   get bounding boxes coordinates and write into array        
            bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
        #   pass the array of bounding boxes coordinates to the imgaug library
            bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
        #   apply augmentation on image and on the bounding boxes
            image_aug, bbs_aug = height_resize(image=image, bounding_boxes=bbs)
        #   write augmented image to a file
            imageio.imwrite(aug_images_path+image_prefix+filename, image_aug)  
        #   create a data frame with augmented values of image width and height
            info_df = group_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)        
            for index, _ in info_df.iterrows():
                info_df.at[index, 'width'] = image_aug.shape[1]
                info_df.at[index, 'height'] = image_aug.shape[0]
        #   rename filenames by adding the predifined prefix
            info_df['filename'] = info_df['filename'].apply(lambda x: image_prefix+x)
        #   create a data frame with augmented bounding boxes coordinates using the function we created earlier
            bbs_df = bbs_obj_to_df(bbs_aug)
        #   concat all new augmented info into new data frame
            aug_df = pd.concat([info_df, bbs_df], axis=1)
        #   append rows to aug_bbs_xy data frame
            aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df])
            
    #   if image width is greater than image height 
    #   AND greater than 600px perform resizing augmentation shrinking image width to 600px
        elif group_df['width'].unique()[0] > group_df['height'].unique()[0] and group_df['width'].unique()[0] > 600:
        #   read the image
            image = imageio.imread(images_path+filename, pilmode="RGB")
        #   get bounding boxes coordinates and write into array        
            bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
        #   pass the array of bounding boxes coordinates to the imgaug library
            bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
        #   apply augmentation on image and on the bounding boxes
            image_aug, bbs_aug = width_resize(image=image, bounding_boxes=bbs)
        #   write augmented image to a file
            imageio.imwrite(aug_images_path+image_prefix+filename, image_aug)  
        #   create a data frame with augmented values of image width and height
            info_df = group_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)        
            for index, _ in info_df.iterrows():
                info_df.at[index, 'width'] = image_aug.shape[1]
                info_df.at[index, 'height'] = image_aug.shape[0]
        #   rename filenames by adding the predifined prefix
            info_df['filename'] = info_df['filename'].apply(lambda x: image_prefix+x)
        #   create a data frame with augmented bounding boxes coordinates using the function we created earlier
            bbs_df = bbs_obj_to_df(bbs_aug)
        #   concat all new augmented info into new data frame
            aug_df = pd.concat([info_df, bbs_df], axis=1)
        #   append rows to aug_bbs_xy data frame
            aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df])

    #     append image info without any changes if it's height and width are both less than 600px 
        else:
            aug_bbs_xy = pd.concat([aug_bbs_xy, group_df])
    # return dataframe with updated images and bounding boxes annotations 
    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)
    return aug_bbs_xy


# In[51]:


# apply resizing augmentation to our images and write the updated images and bounding boxes annotations to the DataFrame 
# we will not apply prefix to our files and will overwrite images in the same directory
resized_images_df = resize_imgaug(labels_df, '1119_images_final_test/', '1119_images_final_test2/', '')


# In[52]:


print(resized_images_df)


# In[53]:


# overwrite the labels.csv with updated info
resized_images_df.to_csv('test_labels.csv', index=False)


# In[55]:


grouped = resized_images_df.groupby('filename')
group_df = grouped.get_group('gate_174.jpg')
group_df = group_df.reset_index()
group_df = group_df.drop(['index'], axis=1)
bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
image = imageio.imread('1119_images_final_test2/gate_174.jpg')
bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
ia.imshow(bbs.draw_on_image(image, size=2))


# In[96]:


# This setup of augmentation parameters will pick two of four given augmenters and apply them in random order
aug = iaa.SomeOf(2, [    
    iaa.Affine(scale=(0.5, 1.5)),
    iaa.Affine(rotate=(-20, 20)),
    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
    iaa.Fliplr(1),
    iaa.Multiply((0.5, 1.5)),
    iaa.GaussianBlur(sigma=(1.0, 3.0)),
    iaa.AdditiveGaussianNoise(scale=(0.03*255, 0.05*255))
])


# In[57]:


def image_aug(df, images_path, aug_images_path, image_prefix, augmentor):
    # create data frame which we're going to populate with augmented image info
    aug_bbs_xy = pd.DataFrame(columns=
                              ['filename','width','height','class', 'xmin', 'ymin', 'xmax', 'ymax']
                             )
    grouped = df.groupby('filename')
    
    for filename in df['filename'].unique():
    #   get separate data frame grouped by file name
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)   
    #   read the image
    #   had to add pilmode="RGB" because error of alpha channel
        image = imageio.imread(images_path+filename, pilmode="RGB")
    #   get bounding boxes coordinates and write into array        
        bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
    #   pass the array of bounding boxes coordinates to the imgaug library
        bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
    #   apply augmentation on image and on the bounding boxes
        image_aug, bbs_aug = augmentor(image=image, bounding_boxes=bbs)
    #   disregard bounding boxes which have fallen out of image pane    
        bbs_aug = bbs_aug.remove_out_of_image()
    #   clip bounding boxes which are partially outside of image pane
        bbs_aug = bbs_aug.clip_out_of_image()
        
    #   don't perform any actions with the image if there are no bounding boxes left in it    
        if re.findall('Image...', str(bbs_aug)) == ['Image([]']:
            pass
        
    #   otherwise continue
        else:
        #   write augmented image to a file
            imageio.imwrite(aug_images_path+image_prefix+filename, image_aug)  
        #   create a data frame with augmented values of image width and height
            info_df = group_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)    
            for index, _ in info_df.iterrows():
                info_df.at[index, 'width'] = image_aug.shape[1]
                info_df.at[index, 'height'] = image_aug.shape[0]
        #   rename filenames by adding the predifined prefix
            info_df['filename'] = info_df['filename'].apply(lambda x: image_prefix+x)
        #   create a data frame with augmented bounding boxes coordinates using the function we created earlier
            bbs_df = bbs_obj_to_df(bbs_aug)
        #   concat all new augmented info into new data frame
            aug_df = pd.concat([info_df, bbs_df], axis=1)
        #   append rows to aug_bbs_xy data frame
            aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df])            
    
    # return dataframe with updated images and bounding boxes annotations 
    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)
    return aug_bbs_xy


# In[130]:


# Apply augmentation to our images and save files into 'aug_images/' folder with 'aug' prefix.
# Write the updated images and bounding boxes annotations to the augmented_images_df dataframe.

# I think depending on the severity of the augment, there might not be a bounding box anymore, therefore it doesn't write it to the new folder
augmented_images_df = image_aug(resized_images_df, '1119_images_final_test2/', 'aug_1119_images_final_test/images/', 'aug_', aug)


# In[143]:


resized_images_df


# In[142]:


augmented_images_df


# In[132]:


grouped_resized = resized_images_df.groupby('filename')
grouped_augmented = augmented_images_df.groupby('filename')

for filename in resized_images_df['filename'].unique():
    
    if filename == "path_486.jpg":

        group_r_df = grouped_resized.get_group(filename)
        group_r_df = group_r_df.reset_index()
        group_r_df = group_r_df.drop(['index'], axis=1)
        bb_r_array = group_r_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
        resized_img = imageio.imread('1119_images_final_test2/'+filename, pilmode="RGB")
        bbs_r = BoundingBoxesOnImage.from_xyxy_array(bb_r_array, shape=resized_img.shape)

        group_a_df = grouped_augmented.get_group('aug_'+filename)
        group_a_df = group_a_df.reset_index()
        group_a_df = group_a_df.drop(['index'], axis=1)
        bb_a_array = group_a_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
        augmented_img = imageio.imread('aug_1119_images_final_test/images/'+'aug_'+filename, pilmode="RGB")
        bbs_a = BoundingBoxesOnImage.from_xyxy_array(bb_a_array, shape=augmented_img.shape)

        ia.imshow(np.hstack([
                bbs_r.draw_on_image(resized_img, size=2),
                bbs_a.draw_on_image(augmented_img, size=2)
                ]))


# In[187]:


grouped_resized = resized_images_df.groupby('filename')
grouped_augmented = augmented_images_df.groupby('filename')

for filename in resized_images_df['filename'].unique():
    print(filename)
    group_r_df = grouped_resized.get_group(filename)
    group_r_df = group_r_df.reset_index()
    group_r_df = group_r_df.drop(['index'], axis=1)
    bb_r_array = group_r_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
    print(bb_r_array)
    resized_img = imageio.imread('1119_images_final_test2/'+filename, pilmode="RGB")
    bbs_r = BoundingBoxesOnImage.from_xyxy_array(bb_r_array, shape=resized_img.shape)

    group_a_df = grouped_augmented.get_group('aug_'+filename)
    group_a_df = group_a_df.reset_index()
    group_a_df = group_a_df.drop(['index'], axis=1)
    bb_a_array = group_a_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
    print(bb_a_array)
    augmented_img = imageio.imread('aug_1119_images_final_test/images/'+'aug_'+filename, pilmode="RGB")
    bbs_a = BoundingBoxesOnImage.from_xyxy_array(bb_a_array, shape=augmented_img.shape)

    ia.imshow(np.hstack([
                bbs_r.draw_on_image(resized_img, size=2),
                bbs_a.draw_on_image(augmented_img, size=2)
                ]))


# In[176]:


# augmented_images_df = augmented_images_df.reset_index()
# augmented_images_df = augmented_images_df.drop("level_0", 1)
# augmented_images_df = augmented_images_df.drop("index", 1)
# augmented_images_df.set("index")
augmented_images_df = augmented_images_df.dropna(how='any',axis=0) 
augmented_images_df


# In[194]:


resized_images_df


# In[173]:


df1 = augmented_images_df[augmented_images_df.isna().any(axis=1)]
df1 = df1.dropna(how='any',axis=0) 
df1


# In[178]:


augmented_images_df.to_csv('augmented.csv', index=False)


# In[179]:


all_labels_df = pd.concat([resized_images_df, augmented_images_df])
all_labels_df.to_csv('all_labels.csv', index=False)


# In[195]:


resized_dict = resized_images_df.to_dict('index')
print(type(resized_dict[0]['filename']))
resized_dict 


# In[180]:


augmented_dict = augmented_images_df.to_dict('index')
print(type(augmented_dict[0]['filename']))
augmented_dict


# In[196]:


resized_dict_values = resized_dict.values()
resized_dict_values


# In[181]:


augmented_dict_values = augmented_dict.values()
augmented_dict_values


# In[182]:


get_ipython().system('ls')


# In[150]:


def class_to_index(class_name):
    if class_name == "small_gate":
        return 0
    elif class_name == "big_gate":
        return 1
    elif class_name == "total_gate":
        return 2
    elif class_name == "path":
        return 3


# In[192]:


def yolo_to_coord(x, y, width, height, width_i, height_i):
    img_w = width_i
    img_h = height_i
    x2, y2 = int((x + width/2)*img_w), int((y + height/2)*img_h)
    x1, y1 = int((x - width/2)*img_w), int((y - height/2)*img_h)
    return x1, y1, x2, y2


# In[201]:


dir_aug = "aug_1119_images_final_test"
for key, ke in zip(augmented_dict_values, resized_dict_values):
    print(key['filename'])
#     key_name = os.path.splitext(key['filename'])[0] + ".txt"
#     txt_file_path = os.path.join(dir_aug, key_name)
#     txt_file = open(txt_file_path, 'a')
#     txt_file.write("test\n")
    class_name = key['class']
    class_val = class_to_index(class_name)
    width = int(key['width'])
    height = int(key['height'])
    x1, y1, x2, y2 = int(key['xmin']), int(key['ymin']), int(key['xmax']), int(key['ymax'])
    x_c = (x1 + x2)/2
    y_c = (y1 + y2)/2
    w = x2 - x1
    h = y2 - y1
    x_c /= width
    y_c /= height
    w /= width
    h /= height
    
    widthq = int(ke['width'])
    heightq = int(ke['height'])
    x1q, y1q, x2q, y2q = int(ke['xmin']), int(ke['ymin']), int(ke['xmax']), int(ke['ymax'])
    x_cq = (x1q + x2q)/2
    y_cq = (y1q + y2q)/2
    wq = x2q - x1q
    hq = y2q - y1q
    x_cq /= widthq
    y_cq /= heightq
    wq /= widthq
    hq /= heightq
    
#     if key['filename'] == "aug_path_486.jpg":
    print("{} {} {} {} \t {} {} {} {}".format(x1, y1, x2, y2, x1q, y1q, x2q, y2q))
    print("{} {} {} {} {} \n{} {} {} {} {} ".format(class_val, x_c, y_c, w, h, class_val, x_cq, y_cq, wq, hq))
    q, w, e, r = yolo_to_coord(x_c, y_c, w, h, width, height)
    a, s, d, f = yolo_to_coord(x_cq, y_cq, wq, hq, widthq, heightq)
    print("{} {} {} {} \t {} {} {} {}".format(q, w, e, r, a, s, d, f))


# In[189]:


q,w,e,r = yolo_to_coord(0.29528795811518327, 0.36666666666666664, 0.5465968586387434, 0.7333333333333333, 955, 600)

print(q)
print(w)
print(e)
print(r)


# In[184]:


len(augmented_dict_values)


# In[203]:


dir_aug = "aug_1119_images_final_test/labels"
for key in augmented_dict_values:
#     print(key['filename'])
    key_name = os.path.splitext(key['filename'])[0] + ".txt"
    txt_file_path = os.path.join(dir_aug, key_name)
    txt_file = open(txt_file_path, 'a')
    class_name = key['class']
    class_val = class_to_index(class_name)
    width = int(key['width'])
    height = int(key['height'])
    x1, y1, x2, y2 = int(key['xmin']), int(key['ymin']), int(key['xmax']), int(key['ymax'])
    x_c = (x1 + x2)/2
    y_c = (y1 + y2)/2
    w = x2 - x1
    h = y2 - y1
    x_c /= width
    y_c /= height
    w /= width
    h /= height
#     print("{} {} {} {} {}".format(class_val, x_c, y_c, w, h))
#     if key['filename'] ==
    txt_file.write("{} {} {} {} {}\n".format(class_val, x_c, y_c, w, h))


# In[ ]:




