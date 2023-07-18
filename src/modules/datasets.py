import os
import numpy as np
import cv2
import tf2_ros
import rospy

from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Image, CompressedImage

class Datasets(object):

    # Initialization
    def __init__(self, dataset):
        self.dataset = dataset

        self._bridge = CvBridge()
        self._tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self._tfBuffer)

    @staticmethod
    def decode_image_rgb_from_unity(unity_img):
        np_arr = np.frombuffer(unity_img, np.uint8)
        im = cv2.imdecode(np_arr, -1)
        img_rgb = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        return img_rgb

    @staticmethod
    def decode_image_depth_from_unity(unity_img):
        buf = np.ndarray(shape=(1, len(unity_img)),
                         dtype=np.uint8, buffer=unity_img)
        img_depth = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        img_depth = np.divide(img_depth, 65535.0)

        return img_depth

    def get_image_msgs_type(self):
       
        if self.dataset == "RobotAtVirtualHome":
            return CompressedImage, CompressedImage # rgb, depth
        elif self.dataset == "Giraff":
            return CompressedImage, Image
        else:
            return Image, Image


    def preprocess_images(self, rgb_msg, depth_msg, depth_range_max):

        if self.dataset == "RobotAtVirtualHome":
            img_rgb = self.decode_image_rgb_from_unity(rgb_msg.data)
            img_depth = self.decode_image_depth_from_unity(depth_msg.data)
            img_depth *= depth_range_max

        elif self.dataset == "RobotAtHome":
            img_rgb = self._bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            img_depth = self._bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            img_rgb = cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_depth = np.divide(cv2.rotate(img_depth, cv2.ROTATE_90_COUNTERCLOCKWISE), 65535.0)
            img_depth *= depth_range_max

        elif self.dataset == "Giraff":
            img_rgb = self.decode_image_rgb_from_unity(rgb_msg.data)
            img_depth = self._bridge.imgmsg_to_cv2(depth_msg)
            img_depth = np.clip(img_depth, 0, 10.0)
            img_depth = img_depth * 65535/10.0
            img_depth = np.array(img_depth, dtype = np.uint16)
            img_depth = np.divide(img_depth, 65535.0)
            img_depth *= depth_range_max

        elif self.dataset == "OpenLORIS":
            img_rgb = self._bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            img_depth = self._bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            img_depth = img_depth.astype(np.float32)
            img_depth = img_depth * 0.0001
            img_depth *= depth_range_max

        elif self.dataset == "uHumans2":
            img_rgb = self._bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            img_depth = self._bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        elif self.dataset == "VirtualGallery":
            img_rgb = self._bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            img_depth = np.divide(self._bridge.imgmsg_to_cv2(depth_msg, "32FC1"), 100.0)

        elif self.dataset == "rio10":
            img_rgb = self._bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            img_depth = np.divide(self._bridge.imgmsg_to_cv2(depth_msg, "16UC1"), 1000.0)

        elif self.dataset == "RobotAtVirtualHome_GT": 
            img_rgb = self._bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            img_depth = self._bridge.imgmsg_to_cv2(depth_msg, "64FC1")

        elif self.dataset == "TUM":
            img_rgb = self._bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            img_depth = self._bridge.imgmsg_to_cv2(depth_msg, "32FC1")

            #mask = np.isnan(img_depth)
            #img_depth[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), img_depth[~mask])


        else:
            img_rgb = self._bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            # Note that img_depth requires to be processed and normalized in range [0.0-1.0]
            img_depth = self._bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            img_depth = np.divide(img_depth, 65535.0)
            img_depth *= depth_range_max

        return img_rgb, img_depth

    def preprocess_img_seg(self, seg_msg):

        if self.dataset == "rio10" or self.dataset == "RobotAtVirtualHome_GT":
            img_seg = self._bridge.imgmsg_to_cv2(seg_msg)
        else:
            img_seg = self._bridge.imgmsg_to_cv2(seg_msg, "rgb8")

        if self.dataset == "uHumans2":
            walls_mask = np.isin(cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY), [82]) # 82: wall, 
        elif self.dataset == "VirtualGallery":
            walls_mask = np.isin(cv2.cvtColor(img_seg, cv2.COLOR_RGB2GRAY), [160, 176, 38]) # 160: wall, 176: doors, 38: paintings, 80: misc
        elif self.dataset == "rio10" or self.dataset == "RobotAtVirtualHome_GT":
            walls_mask = np.isin(img_seg, [255])

        return walls_mask

    def get_camera_extrinsics(self, image_msg):

        
        if self.dataset == "Giraff":
            camera_robot_transform = self._tfBuffer.lookup_transform('base_link',
                                                                            "camera_down_link",
                                                                            rospy.Time())
            camera_robot_transform.transform.rotation = Quaternion(0.0242788, -0.0703922, 0.0242788, 0.9969283)
            camera_robot_transform.transform.translation.x = -0.04
            camera_robot_transform.transform.translation.z = 0.9
        elif self.dataset == "uHumans2":
            camera_robot_transform = self._tfBuffer.lookup_transform('base_link_gt',
                                                                        image_msg.header.frame_id,
                                                                        rospy.Time())
            camera_robot_transform.transform.rotation = Quaternion(0., 0., 0., 1.)
        elif self.dataset == "RobotAtHome":
            camera_robot_transform.transform.rotation = Quaternion(0.0, 0.0, 0.3826834, 0.9238795)
        elif self.dataset == "TUM":
            camera_robot_transform = self._tfBuffer.lookup_transform('kinect',
                                                                        "openni_camera",
                                                                        rospy.Time())
            #camera_robot_transform.transform.translation.x = 0.
            #camera_robot_transform.transform.translation.y = 0.
            #camera_robot_transform.transform.translation.z = 0.
            #camera_robot_transform.transform.rotation = Quaternion(0., 0., 0., 1.)
        elif self.dataset == "OpenLORIS":
            camera_robot_transform = self._tfBuffer.lookup_transform('base_link',
                                                                            image_msg.header.frame_id,
                                                                            rospy.Time())
            camera_robot_transform.transform.rotation = Quaternion(0.0033077, 0.0080805, 0.0049632, 0.9999496)
        else:
            camera_robot_transform = self._tfBuffer.lookup_transform('base_link',
                                                                            image_msg.header.frame_id,
                                                                            rospy.Time())
        

        return camera_robot_transform
