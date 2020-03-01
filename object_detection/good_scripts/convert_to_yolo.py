## if you want to test with an image:
import cv2
def convert_to_yolo_no_image(img_path, x1, y1, x2, y2):
    """ Returns normalized coordinates for yolo annotation

    Parameters:
        img_path(str): Path to the image file
        x1, y1, x2, y2 (int): The four corners of the bounding box with preferably x2 > x1, y2 > y1

    Returns:
        x_center, y_center, rel_width, rel_height (int): Normalized/relative values rel_x_center_bounding box, rel_y_center_bounding_box, rel_width_bounding_box, rel_height_bounding_box
    """
    image = cv2.imread(img_path)
    if not image:
        raise Exception("Image at {} does not exist".format(img_path))
    image_width = image.shape[1]
    image_height = image.shape[0]
    x_center = (x1+x2)/2
    x_center /= image_width
    y_center = (y1+y2)/2
    y_center / image_height
    rel_width = abs(x2-x1)/image_width
    rel_height = abs(y2-y1)/image_height
    return x_center, y_center, rel_width, rel_height

# # No image
# def convert_to_yolo_old(x1, y1, bb_width, bb_height, img_width, img_height):
#     """ Returns normalized coordinates for yolo annotation

#     Parameters:
#         img_path(str): Path to the image file
#         x1, y1, x2, y2 (int): The four corners of the bounding box with preferably x2 > x1, y2 > y1

#     Returns:
#         x_center, y_center, rel_width, rel_height (int): Normalized/relative values rel_x_center_bounding box, rel_y_center_bounding_box, rel_width_bounding_box, rel_height_bounding_box
#     """
#     x_center = (x1+bb_width/2)/img_width
#     y_center = (y1+bb_height/2)/img_height
#     rel_width = bb_width/img_width
#     rel_height = bb_height/img_height
#     return x_center, y_center, rel_width, rel_height

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