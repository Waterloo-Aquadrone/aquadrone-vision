import os, json
import pandas as pd

def json_to_csv(labels_dir, output_dir, output_name):

    if not os.path.isdir(labels_dir):
        raise Exception("Label directory: {} is not a directory".format(labels_dir))
    elif not os.path.isdir(output_dir):
        raise Exception("Output directory: {} is not a directory".format(labels_dir))

    output_name = str(output_name)

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
                width_list.append(key['width'])
                height_list.append(key['height'])
            elif len(key) != 0:
                dict_obj = key[0]
                class_name = dict_obj['classTitle']
                class_list.append(class_name)
                
                box_dict = dict_obj['points']
                box_list = box_dict['exterior']
                x1, y1 = box_list[0]
                x2, y2 = box_list[1]
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

    data_labels = {'filename': file_name_list, 'width': width_list, 'height': height_list, 
            'class': class_list, 'xmin': xmin_list, 'ymin': ymin_list, 'xmax':xmax_list, 'ymax':ymax_list}

    labels_df = pd.DataFrame(data = data_labels)
    labels_df.to_csv(os.path.join(output_dir, output_name + ".csv"), index=None)
    print("Succesfully converted json files to {}".format(output_name + ".csv"))
    
json_to_csv("train_valid/train_labels", ".", "test_csv")