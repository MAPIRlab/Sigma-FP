#! /usr/bin/env python

import os
import sys
import time

import cv2
import random
import numpy as np
from math import pi
import open3d as o3d
from copy import deepcopy

import rospy
import tf2_ros
import message_filters
from cv_bridge import CvBridge, CvBridgeError

from sklearn.cluster import DBSCAN

from std_msgs.msg import Header, String, ColorRGBA
from geometry_msgs.msg import PoseWithCovarianceStamped, Transform, TransformStamped, Quaternion
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from detectron2_ros.msg import ResultWithWalls
from visualization_msgs.msg import MarkerArray
from sigmafp.msg import WallMeshArray

from utils import PlaneManager, Transformations

class FloorplanReconstruction(object):
    def __init__(self):

        rospy.logwarn("Initializing Sigma-FP: 3D Floorplan Reconstruction")

        # ROS Parameters
        self.image_rgb_topic = self.load_param('~topic_cameraRGB', "ViMantic/virtualCameraRGB")
        self.image_depth_topic = self.load_param('~topic_cameraDepth', "ViMantic/virtualCameraDepth")
        self.semantic_topic = self.load_param('~topic_result', 'ViMantic/Detections')
        self.cnn_topic = self.load_param('~topic_cnn', 'detectron2_ros/result')
        self.image_toCNN = self.load_param('~topic_republic', 'ViMantic/ToCNN')
        self.dataset = self.load_param('~dataset', "RobotAtVirtualHome")
        self.debug = self.load_param('~debug', False)

        # Camera Calibration
        self._width = self.load_param('~image_width', 640)
        self._height = self.load_param('~image_height', 480)
        self._cx = self.load_param('~camera_cx', 320)
        self._cy = self.load_param('~camera_cy', 240)
        self._fx = self.load_param('~camera_fx', 304.0859)
        self._fy = self.load_param('~camera_fy', 312.7741)
        self._depth_range_max = self.load_param('~camera_depth_max_range', 10.0)
        self._min_reliable_cam_depth = self.load_param('~min_reliable_depth_value', 0.01)
        self._max_reliable_cam_depth = self.load_param('~max_reliable_depth_value', self._depth_range_max - 0.01)

        # Flow control variables, flags and counters
        self._publish_rate = 10             # Rate of ROS publishing
        self._max_tries = 10                # Max tries for the CNN to process an image before skipping the image
        self._tries = 0                     # Number of current tries to get the CNN available to process the image
        self._image_counter = 0             # Number of images processed by the CNN
        self._n_iterations = 0              # Number of iterations of our method
        self._n_iterations_openings = 0     # Number of tries to detect openings
        self._flag_processing = False       # Waiting a new input image to process
        self._flag_cnn = False              # CNN is occupied

        # Timers
        self._start_time = 0                # Time when the code starts
        self._total_time = 0                # Time elapsed until now
        self._time_plane_extraction = 0     # Total time spent in plane extraction
        self._time_opening_detection = 0    # Total time spent in opening detection
        self._time_plane_matching = 0       # Total time spent in plane matching

        # Image variables
        self._last_msg = None               # Last input image received
        self._last_cnn_result = None        # Last output of the CNN
        self._image_r = None                # Rows grid (cx - row_idx) / fx
        self._image_c = None                # Columns grid (cy - column_idx) / fy

        # Parameters and variables of Sigma-FP
        self._n_points_in_pcd = self.load_param('~points_in_pcd', 6000)  # Number of points in the downgraded point cloud
        self._min_points_plane = self.load_param('~min_points_plane', 60)  # Minimum number of points to consider a planar patch as a possible wall  # 100
        self._min_plane_width = self.load_param('~min_plane_width', 0.35)  # Minimum width to accept a planar patch as a possible wall (in meters)
        self._min_px_opening = self.load_param('~min_px_opening', 7000)  # Minimum number of pixels to consider a region as a opening  # 8000
        self._bhattacharyya_threshold = self.load_param('~bhattacharyya_threshold', 10)  # Threshold for the statistical distance of Bhattacharyya  # real 7
        self._euclidean_threshold = self.load_param('~euclidean_threshold', 1.1)  # Threshold for the minimum euclidean distance between walls  # 0.3
        self._eps_alpha = self.load_param('~eps_alpha', 8.0) * pi / 180.0  # Epsilon for DBSCAN of the azimuth angle of the plane (in radians)  # 1
        self._eps_beta = self.load_param('~eps_beta', 8.0) * pi / 180.0  # Epsilon for DBSCAN of the elevation angle of the plane (in radians) # 10
        self._eps_dist = self.load_param('~eps_dist', 0.005)  # Epsilon for DBSCAN of the plane-to-origin distance (in meters)  # real 0.02
        self._num_walls = 0  # Number of current detected walls
        self._walls = {}  # Dictionary including the complete information of the current floorplan

        # Transformations
        self._camera_robot_transform = None     # Relative pose of the camera frame w.r.t. the robot frame

        # Publishers
        self._pub_processed_image = rospy.Publisher(self.image_toCNN, Image, queue_size=1)
        self._walls_pub = rospy.Publisher('walls', MarkerArray, queue_size=10)
        self._pcd_pub = rospy.Publisher('pointcloud_visualization', PointCloud2, queue_size=1)
        self._unity_pub = rospy.Publisher('wall_mesh', WallMeshArray, queue_size=10)

        # Subscribers
        rospy.Subscriber(self.cnn_topic, ResultWithWalls, self.callback_new_detection)
        rospy.Subscriber("wallmap_commands", String, self.callback_commands)

        if self.dataset == "RobotAtVirtualHome":
            sub_rgb_image = message_filters.Subscriber(self.image_rgb_topic, CompressedImage)
            sub_depth_image = message_filters.Subscriber(self.image_depth_topic, CompressedImage)
        elif self.dataset == "Giraff":
            sub_rgb_image = message_filters.Subscriber(self.image_rgb_topic, CompressedImage)
            sub_depth_image = message_filters.Subscriber(self.image_depth_topic, Image)
        else:
            sub_rgb_image = message_filters.Subscriber(self.image_rgb_topic, Image)
            sub_depth_image = message_filters.Subscriber(self.image_depth_topic, Image)

        sub_pose_amcl = message_filters.Subscriber('amcl_pose', PoseWithCovarianceStamped)
        message_filter = message_filters.ApproximateTimeSynchronizer([sub_depth_image, sub_rgb_image, sub_pose_amcl],
                                                                     10, 0.1)
        message_filter.registerCallback(self.callback_synchronize_image)

        # Handlers
        self._bridge = CvBridge()
        self._tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self._tfBuffer)
        self._tr = Transformations()
        self._pm = PlaneManager(self._height, self._width, self._cx, self._cy, self._fx, self._fy,
                                self._depth_range_max, self._min_px_opening)

        rospy.logwarn("Initialized")

    ####################################################################################################################
    ################################################### Node Script ####################################################
    ####################################################################################################################

    def run(self):

        rate = rospy.Rate(self._publish_rate)

        while not rospy.is_shutdown():
            # Extracting and projecting detected walls
            if self._flag_processing and self._flag_cnn:

                # Getting the mask of pixels belonging to walls
                try:
                    wall_mask = self._bridge.imgmsg_to_cv2(self._last_cnn_result.walls) == 255
                except CvBridgeError as e:
                    print(e)
                    continue

                # Check if there are pixels belonging to walls in the image
                if not np.max(np.max(wall_mask)):
                    self._flag_cnn = False
                    self._flag_processing = False
                    continue

                # Obtain 3D coordinates, in meters, of each pixel
                z = self._last_msg[2] * self._depth_range_max
                x = self._image_c * z
                y = self._image_r * z

                time_start = time.time()

                # Create point cloud of walls
                point_cloud = np.array([z[wall_mask].reshape(-1), x[wall_mask].reshape(-1), y[wall_mask].reshape(-1)]).T

                # Reduce the point cloud to the data in the reliable range of the depth sensor
                point_cloud = point_cloud[np.logical_and(point_cloud[:, 0] > self._min_reliable_cam_depth,
                                                         point_cloud[:, 0] < self._max_reliable_cam_depth)].copy()

                if self.dataset == "Giraff":
                    point_cloud = point_cloud[point_cloud[:, 2] < 1.5].copy()

                # Create the point cloud in open3d
                pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point_cloud))

                # Downsampling point cloud
                try:
                    pcd = pcd.uniform_down_sample(int(len(point_cloud) / self._n_points_in_pcd))
                except:
                    pass

                pcd_np = np.asarray(pcd.points)

                print("Number of points in downsampled pointcloud: {}/{}".format(pcd_np.shape[0], len(point_cloud)))

                if pcd_np.shape[0] == 0:
                    self._flag_cnn = False
                    self._flag_processing = False
                    continue

                # Obtain mean distance between points in the point cloud
                # More precise but very time-consuming (Not recommended)
                #tree = KDTree(pcd_np)
                #dist, _ = tree.query(pcd_np, k=100)
                #mean_dist_pcd = np.sum(dist[:, 99]) / dist.shape[0]

                # To avoid high consumption times, please use the following: (Recommended)
                mean_dist_pcd = 0.1

                # Compute, normalize and orient normals towards camera location
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=mean_dist_pcd, max_nn=100))
                pcd.normalize_normals()
                pcd.orient_normals_towards_camera_location()

                # Create transform msg from robot to map
                trans_amcl = TransformStamped()
                trans_amcl.header = Header(0, rospy.Time(), 'map')
                trans_amcl.child_frame_id = "base_link"
                trans_amcl.transform = Transform(self._last_msg[5].pose.pose.position,
                                                 self._last_msg[5].pose.pose.orientation)

                # Obtain transformation matrixes
                tr_matrix_robot_camera = self._tr.msg_to_se3(self._camera_robot_transform)  # Robot -> Camera
                tr_matrix_robot_map = self._tr.msg_to_se3(trans_amcl)                       # Robot -> Map
                tr_matrix_camera_map = np.matmul(tr_matrix_robot_map, tr_matrix_robot_camera)   # Camera -> Map
                inv_tr_matrix = np.linalg.inv(tr_matrix_camera_map)

                # Obtain point cloud wrt robot frame
                pcd = pcd.transform(tr_matrix_robot_camera)
                pcd_np = np.asarray(pcd.points)

                # Obtain point cloud in global coordinates wrt world frame
                pcd_global = deepcopy(pcd)
                pcd_global.transform(tr_matrix_robot_map)

                # Debugging: show in rviz the processed point cloud
                if self.debug:
                    self._pcd_pub.publish(self._pm.point_cloud_visualization_rviz(pcd_global, self._last_msg[0]))

                # Generate data for clustering: elevation and azimuth angles from normals and distance plane-origin
                normals = np.array(pcd.normals)
                alpha = np.arctan2(normals[:, 1], normals[:, 0]).reshape(-1, 1)
                beta = np.arccos(normals[:, 2]).reshape(-1, 1)
                distance_to_origin = np.sum(normals * pcd_np, axis=1).reshape(-1, 1)
                dbscan_data = np.concatenate((alpha, beta, distance_to_origin), axis=1)
                dbscan_data_scaled = dbscan_data / np.asarray([self._eps_alpha, self._eps_beta, self._eps_dist])

                # DBSCAN Clustering
                clustering = DBSCAN(eps=1., min_samples=self._min_points_plane).fit(dbscan_data_scaled)
                labels = clustering.labels_.astype(np.float_)
                max_label = labels.max()

                # Time for plane extraction
                self._time_plane_extraction += time.time() - time_start

                # Characterizing each clustered wall by: its Gaussian distribution and a set of features
                for idx in range(max_label.astype(np.int_) + 1):

                    # Obtaining the set of points belonging to the specific cluster
                    wall_pps = dbscan_data[np.where(labels == idx)[0]]

                    # Computing mean and covariance of the cluster
                    mean_pps = np.mean(wall_pps, axis=0).reshape((3, 1))
                    cov_pps = np.cov(wall_pps.T)

                    # Skipping walls that do not meet the Atlanta world assumption
                    if abs((pi / 2) - mean_pps[1]) > 0.25:
                        continue

                    # Changing the reference system of the Gaussian distribution: from robot to world frame
                    mean_global, cov_global = self._pm.pps_from_robot_to_map(self._last_msg[5], mean_pps, cov_pps)

                    # Too much uncertainty in the robot localization... Skipping data
                    if mean_global is None:
                        rospy.logwarn("Bad localization, skipping data...")
                        continue

                    # Creating a dictionary with the characterization of the wall
                    wall_dict = {"mean": mean_global.reshape((3,)), "cov": cov_global}

                    # Obtaining the point cloud of the wall in Cartesian space
                    wall_pcd = pcd_global.select_by_index(np.where(labels == idx)[0])
                    _, inliers = wall_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                    wall_pcd = wall_pcd.select_by_index(inliers)

                    # Remaining features of the wall
                    wall_dict["n_samples"] = np.asarray(wall_pcd.points).shape[0]
                    wall_dict["n_detections"] = 1
                    wall_dict["max_bound"] = wall_pcd.get_max_bound()
                    wall_dict["min_bound"] = wall_pcd.get_min_bound()
                    wall_dict["color"] = ColorRGBA(random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1), 1.0)
                    wall_dict["first_seen"] = rospy.Time.now().to_sec()

                    plane_width = np.sqrt((wall_dict["max_bound"][0] - wall_dict["min_bound"][0]) ** 2 +
                                          (wall_dict["max_bound"][1] - wall_dict["min_bound"][1]) ** 2)

                    plane_height = wall_dict["max_bound"][2] - wall_dict["min_bound"][2]

                    # Extracting openings in walls
                    opening_start = time.time()
                    wall_dict["openings"] = self._pm.detect_openings_in_plane(wall_dict, inv_tr_matrix,
                                                                 self._last_msg[1], self._last_msg[2])

                    self._time_opening_detection += time.time() - opening_start
                    self._n_iterations_openings += 1

                    # Accepting a wall if it has a minimum width... otherwise it is considered as non-informative
                    if plane_width > self._min_plane_width and plane_height > self._min_plane_width:
                        self._walls[str(self._num_walls)] = wall_dict
                        self._num_walls += 1
                    else:
                        rospy.logwarn("Neglected because not sufficient width.")

                # Another clustering approach
                matching_start = time.time()
                order, plane_features = self._pm.plane_dict_to_features_comp(self._walls)

                # Data association and integration process
                new_walls, num_walls = self._pm.match_and_merge_planes(self._walls, plane_features, order,
                                                                       self._pm.bhattacharyya_distance_features,
                                                                       self._bhattacharyya_threshold,
                                                                       self._euclidean_threshold)

                self._time_plane_matching += time.time() - matching_start

                # Updating the map
                self._num_walls = num_walls
                self._walls = {}
                self._walls = new_walls

                # Showing current floorplan
                self._walls_pub.publish(self._pm.create_msg_walls_markers(self._walls))

                # Print time information
                try:
                    loop_time = time.time() - time_start
                    self._total_time += loop_time
                    self._n_iterations += 1
                    print("Loop time: " + str(1000.0*loop_time) + " ms / Mean Loop Time: " +
                          str(1000.0*self._total_time / self._n_iterations) + str(" ms"))
                    print("Average plane extraction time: {:.2f} ms".format(1000.0*self._time_plane_extraction / self._n_iterations))
                    print("Average opening detection time: {:.2f} ms".format(1000.0*self._time_opening_detection / self._n_iterations_openings))
                    print("Average plane matching time: {:.2f} ms".format(1000.0*self._time_plane_matching / self._n_iterations))
                    print("Number of current planes: " + str(self._num_walls))
                except:
                    pass
                ###################################

                self._flag_cnn = False
                self._flag_processing = False

            # CNN does not respond.
            elif self._flag_processing and not self._flag_cnn:
                # Skipping frame.
                if self._tries > self._max_tries:
                    self._flag_processing = False
                    rospy.logwarn("[Sigma-FP] CNN does not respond, skipping frame.")

                # Awaiting CNN to process the last image
                else:
                    self._pub_processed_image.publish(self._bridge.cv2_to_imgmsg(self._last_msg[1], 'rgb8'))
                    self._tries += 1

            rate.sleep()

    ####################################################################################################################
    ##################################################### Callbacks ####################################################
    ####################################################################################################################

    def callback_synchronize_image(self, depth_msg, rgb_msg, pose_msg):
        if not self._flag_processing:

            if self.dataset == "RobotAtVirtualHome":
                img_rgb = self.decode_image_rgb_from_unity(rgb_msg.data)
                img_depth = self.decode_image_depth_from_unity(depth_msg.data)

            elif self.dataset == "RobotAtHome":
                img_rgb = self._bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
                img_depth = self._bridge.imgmsg_to_cv2(depth_msg, "16UC1")
                img_rgb = cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_depth = np.divide(cv2.rotate(img_depth, cv2.ROTATE_90_COUNTERCLOCKWISE), 65535.0)

            elif self.dataset == "Giraff":
                img_rgb = self.decode_image_rgb_from_unity(rgb_msg.data)
                img_depth = self._bridge.imgmsg_to_cv2(depth_msg)
                img_depth = np.clip(img_depth, 0, 10.0)
                img_depth = img_depth * 65535/10.0
                img_depth = np.array(img_depth, dtype = np.uint16)
                img_depth = np.divide(img_depth, 65535.0)

            elif self.dataset == "OpenLORIS":
                img_rgb = self._bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
                img_depth = self._bridge.imgmsg_to_cv2(depth_msg, "16UC1")
                img_depth = img_depth.astype(np.float32)
                img_depth = img_depth * 0.0001

            else:
                img_rgb = self._bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
                # Note that img_depth requires to be processed and normalized in range [0.0-1.0]
                img_depth = np.divide(self._bridge.imgmsg_to_cv2(depth_msg, "16UC1"), 65535.0)

            # Manual configuration of the Camera-Robot transform -> For custom dataset, please set the transform
            # in the _camera_robot_transform variable as done in the examples.
            try:
                if self._camera_robot_transform is None:
                    try:
                        if self.dataset == "Giraff":
                            self._camera_robot_transform = self._tfBuffer.lookup_transform('base_link',
                                                                                           "camera_down_link",
                                                                                           rospy.Time())
                            self._camera_robot_transform.transform.rotation = Quaternion(0.0242788, -0.0703922, 0.0242788, 0.9969283)
                            self._camera_robot_transform.transform.translation.x = -0.04
                            self._camera_robot_transform.transform.translation.z = 0.9
                        else:
                            self._camera_robot_transform = self._tfBuffer.lookup_transform('base_link',
                                                                                           rgb_msg.header.frame_id,
                                                                                           rospy.Time())
                        if self.dataset == "RobotAtHome":
                            self._camera_robot_transform.transform.rotation = Quaternion(0.0, 0.0, 0.3826834, 0.9238795)
                        elif self.dataset == "OpenLORIS":
                            self._camera_robot_transform = self._tfBuffer.lookup_transform('base_link',
                                                                                           depth_msg.header.frame_id,
                                                                                           rospy.Time())
                            self._camera_robot_transform.transform.rotation = Quaternion(0.0033077, 0.0080805, 0.0049632, 0.9999496)

                    except:
                        pass

                self._last_msg = [rgb_msg.header, img_rgb, img_depth, None, None, pose_msg]
                self._pub_processed_image.publish(self._bridge.cv2_to_imgmsg(img_rgb, 'rgb8'))
                self._tries = 0

                if self._start_time == 0:
                    self._start_time = time.time()

                if self._image_c is None and self._image_r is None:
                    # Generate a meshgrid where each pixel contains its pixel coordinates
                    self._height, self._width = img_depth.shape
                    self._image_c, self._image_r = np.meshgrid(np.arange(self._width), np.arange(self._height),
                                                               sparse=True)
                    self._image_c = (self._cx - self._image_c) / self._fx
                    self._image_r = (self._cy - self._image_r) / self._fy

                if self._camera_robot_transform != None:
                    self._flag_processing = True

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Failing to retrieve robot pose")

    def callback_new_detection(self, result_cnn):

        if self._flag_processing and not self._flag_cnn:
            self._image_counter = self._image_counter + 1
            if (self._image_counter % 11) == 10:
                rospy.loginfo("Images detected per second=%.2f",
                              float(self._image_counter) / (time.time() - self._start_time))

            if len(result_cnn.class_names) > 0 or np.max(np.max(self._bridge.imgmsg_to_cv2(result_cnn.walls))) > 0:
                self._last_cnn_result = result_cnn
                self._flag_cnn = True

            else:
                self._flag_processing = False

    # This callback performs the Global Refinement Step. To run it, please publish a message in the /wallmap_commands
    # topic.
    def callback_commands(self, data):

        command = data.data

        if not os.path.exists(command) or command[0] != '/':
            command = "/home/jose/Escritorio/"

            # Another clustering approach
            order, plane_features = self._pm.plane_dict_to_features_comp(self._walls)

            new_walls, num_walls = self._pm.match_and_merge_planes(self._walls, plane_features, order,
                                                                   self._pm.bhattacharyya_distance_features,
                                                                   self._bhattacharyya_threshold,
                                                                   self._euclidean_threshold)

            self._num_walls = num_walls
            self._walls = {}
            self._walls = new_walls

            # Saving Map
            if command[0] == '/':
                dir = command + "wallmap_no_refined" + ".npy"

                try:
                    np.save(dir, self._walls)
                    print("Map saved succesfully in " + dir)
                except:
                    print("Error! Map cannot be saved in the given directory.")

            init_time_refinement = time.time()

            self._walls = self._pm.global_refinement(self._walls)

            print('Time for global refinement: ' + str(time.time() - init_time_refinement) + 'seconds')
            self._walls_pub.publish(self._pm.create_msg_walls_markers(self._walls))

            dir = command + "wallmap_refined" + ".npy"

            try:
                np.save(dir, self._walls)
                print("Refined Map saved succesfully in " + dir)
            except:
                pass

    ####################################################################################################################
    ################################################### Static Methods #################################################
    ####################################################################################################################

    @staticmethod
    def load_param(param, default=None):
        new_param = rospy.get_param(param, default)
        rospy.loginfo("[Sigma-FP] %s: %s", param, new_param)
        return new_param

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


########################################################################################################################
######################################################### Main #########################################################
########################################################################################################################

def main(argv):
    rospy.init_node('3D-Floorplan-Reconstruction')
    node = FloorplanReconstruction()
    node.run()


if __name__ == '__main__':
    main(sys.argv)
