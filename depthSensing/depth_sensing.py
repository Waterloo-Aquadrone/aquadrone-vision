########################################################################
#
# Copyright (c) 2017, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import pyzed.sl as sl
from pixel_depth import depth_value
import math
import numpy as np
import sys
import cv2
import os

def main():
    # Create a Camera object
    zed = sl.Camera()

    # !!! Modify these paths to save images to desired dirs !!!
    path_img = r'C:\Users\Jesse\source\repos\zed-python-api\tutorials\tutorial 3 - depth sensing\images\reg'
    path_depth_img = r'C:\Users\Jesse\source\repos\zed-python-api\tutorials\tutorial 3 - depth sensing\images\depth'

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD # Use STANDARD sensing mode

    # Set image size
    img_size = zed.get_camera_information().camera_resolution
    img_size.width = round(img_size.width / 2)
    img_size.height = round(img_size.height / 2)

    # Create image, depth image and point cloud matrices
    i = 0
    image_zed = sl.Mat(img_size.width, img_size.height, sl.MAT_TYPE.U8_C4)
    depth_img_zed = sl.Mat(img_size.width, img_size.height, sl.MAT_TYPE.U8_C4)
    depth_val_zed = sl.Mat(img_size.width, img_size.height, sl.MAT_TYPE.U8_C4)
    point_cloud = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    tr_np = mirror_ref.m

    while True:
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:

            # Retrieve left image and depth image
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, img_size)
            zed.retrieve_image(depth_img_zed, sl.VIEW.DEPTH, sl.MEM.CPU, img_size)    
            # Gives depth value at a certain pixel
            # Test depth value at center pixel
            ## Add condition to determine when to use, or just remain the same ##
            ## Maybe get center value of bounding box(es) as the dimension w.r.t. the center of image ##
            if True:
                zed.retrieve_measure(depth_val_zed, sl.MEASURE.DEPTH, sl.MEM.CPU, img_size)
                depth_val_ocv = depth_val_zed.get_data()
                # For placeholder, gives value at the center of image
                depth = depth_value(int(len(depth_val_ocv)/2), int(len(depth_val_zed[0])/2), depth_val_ocv)
                print(depth)        
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)       

            # Retrive image Numpy array
            img_ocv = image_zed.get_data()
            depth_img_ocv = depth_img_zed.get_data()
            # Load depth data into Numpy array      

            # Display what the camera sees
            cv2.imshow("Image", img_ocv)
            cv2.imshow("Depth", depth_img_ocv)
            cv2.waitKey(1)
            
            # Get and print distance value in mm at the center of the image
            # We measure the distance camera - object using Euclidean distance
            point_cloud_value = point_cloud.get_value(img_size.width, img_size.height)

            # Cloud[0] = x, Cloud[1] = y, Cloud[2] = z
            x = point_cloud_value[0]
            y = point_cloud_value[1]
            z = point_cloud_value[2]

            # Measure the distance from object -> camera using Euclidean distance
            distance = math.sqrt(x**2 + y**2 + z**2)

            point_cloud_np = point_cloud.get_data()
            point_cloud_np.dot(tr_np)

            if not np.isnan(distance) and not np.isinf(distance):
                distance = round(distance)
                print("x: {0}mm, y: {1}mm, z: {2}mm".format(x, y, z))
                print("Distance to Camera at ({0}, {1}): {2} mm\n".format(img_size.width, img_size.height, distance))
                # Increment the file number
                i += 1
            else:
                print("Can't estimate distance at this position, move the camera\n")
            sys.stdout.flush()

            # save img_ocv to reg dir
            os.chdir(path_img)
            cv2.imwrite('test{0}.jpg'.format(i), img_ocv)

            # save dep_img_ocv to depth dir
            os.chdir(path_depth_img)
            cv2.imwrite('test{0}.jpg'.format(i), depth_img_ocv)    

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
