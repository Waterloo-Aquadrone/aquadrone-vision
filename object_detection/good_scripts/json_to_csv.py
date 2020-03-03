import os, json
import pandas as pd

# for no images
def convert_to_yolo(x1, y1, x2, y2, img_width, img_height):
    """ Returns normalized coordinates for yolo annotation

    Parameters:
        x1, y1, x2, y2 (int): The four corners of the bounding box with preferably x2 > x1, y2 > y1
        img_width (int): width of whole image in pixels
        img_height (int): height of whole image in pixels

    Returns:
        x_center, y_center, rel_width, rel_height (int): Normalized/relative values rel_x_center_bounding box, rel_y_center_bounding_box, rel_width_bounding_box, rel_height_bounding_box
    """
    x_center = (x1+x2)/2
    x_center /= img_width
    y_center = (y1+y2)/2
    y_center /= img_height
    rel_width = (x2-x1)/img_width
    rel_height = (y2-y1)/img_height
    return x_center, y_center, rel_width, rel_height

def json_to_csv(labels_dir, output_dir, output_name, yolo_format = True):

    if not os.path.isdir(labels_dir):
        raise Exception("Label directory: {} is not a directory".format(labels_dir))
    elif not os.path.isdir(output_dir):
        raise Exception("Output directory: {} is not a directory".format(labels_dir))

    output_name = str(output_name)
    if os.path.isfile(os.path.join(output_dir, output_name + ".csv")):
        raise Exception("Output name: {} already exists at path {}".format(output_name, os.path.join(output_dir, output_name + ".csv")))

    # Get list of all file names in labels_dir
    label_list = os.listdir(labels_dir)
    for name in label_list:
        if name.startswith("."):
            os.remove(os.path.join(labels_dir, name))
    # print(len(label_list))
    # print(label_list)

    # create vars to hold info from json files to put into a pandas dataframe
    file_name_list = []
    width_list = []
    height_list = []
    class_list = []
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []

    for name in label_list:
        file_path = os.path.join(labels_dir, name)
        file = open(file_path, 'r')
        # loads json into a dictionary 
        file_data = json.load(file)

        # splits filename into a name and extension as an array, we take the name 
        img_name = os.path.splitext(name)[0]

        # appends img_name to the list of files
        file_name_list.append(img_name)
        for key in file_data.values():
            if isinstance(key, dict):
                img_width = key['width']
                img_height = key['height']
                width_list.append(img_width)
                height_list.append(img_height)
            elif len(key) != 0:
                dict_obj = key[0]
                class_name = dict_obj['classTitle']
                class_list.append(class_name)
                
                box_dict = dict_obj['points']
                box_list = box_dict['exterior']
                x1, y1 = box_list[0]
                x2, y2 = box_list[1]

                if yolo_format:
                    x1, y1, x2, y2 = convert_to_yolo(x1, y1, x2, y2, img_width, img_height)

                xmin_list.append(x1)
                ymin_list.append(y1)
                xmax_list.append(x2)
                ymax_list.append(y2)
        


    # print(len(file_name_list))
    # print(len(width_list))
    # print(len(height_list))
    # print(len(class_list))
    # print(len(xmin_list))
    # print(len(ymin_list))
    # print(len(xmax_list))
    # print(len(ymax_list))

    if yolo_format:
        data_labels = {'filename': file_name_list, 'width': width_list, 'height': height_list, 
                'class': class_list, 'rel_x_center': xmin_list, 'rel_y_center': ymin_list, 'rel_width':xmax_list, 'rel_height':ymax_list}
    else :
        data_labels = {'filename': file_name_list, 'width': width_list, 'height': height_list, 
                'class': class_list, 'xmin': xmin_list, 'ymin': ymin_list, 'xmax':xmax_list, 'ymax':ymax_list}

    labels_df = pd.DataFrame(data = data_labels)
    labels_df.to_csv(os.path.join(output_dir, output_name + ".csv"), index=None)
    print("Succesfully converted json files to {}".format(output_name + ".csv"))
    
# json_to_csv("train_valid/train_labels", "train_valid", "train_test_labels_yolo2", True)
