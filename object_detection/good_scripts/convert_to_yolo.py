import cv2

def convert_to_yolo(img_path, x1, y1, width, height):

    image = cv2.imread(img_path)
    if not image:
        raise Exception("Image at {} does not exist".format(img_path))
    image_width = image.shape[1]
    image_height = image.shape[0]
    x_center = (x1+width/2)/image_width
    y_center = (y1+height/2)/image_height
    rel_width = width/image_width
    rel_height = height/image_height
    return x_center, y_center, rel_width, rel_height