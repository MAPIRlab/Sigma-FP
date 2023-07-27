#! /usr/bin/env python3

import os
import sys
import time

import cv2
import random
import numpy as np
from math import pi
import open3d as o3d
import networkx as nx
from copy import deepcopy
import matplotlib.pyplot as plt

import rospy
import tf2_ros
import message_filters
from cv_bridge import CvBridge, CvBridgeError

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from scipy.stats import circmean

from std_msgs.msg import Header, String, ColorRGBA
from geometry_msgs.msg import PoseWithCovarianceStamped, Transform, TransformStamped, Quaternion
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from detectron2_ros.msg import ResultWithWalls
from visualization_msgs.msg import MarkerArray
from sigmafp.msg import WallMeshArray

from modules.planes import PlaneManager
from modules.transformations import Transformations
from modules.external_transformations import *
from modules.datasaver import DataSaver
from modules.datasets import Datasets


class FloorplanReconstruction(object):
    def __init__(self):

        rospy.logwarn("Initializing Sigma-FP: 3D Floorplan Reconstruction")

        # Topic names
        self.image_rgb_topic = self.load_param('~topic_cameraRGB', "ViMantic/virtualCameraRGB")
        self.image_depth_topic = self.load_param('~topic_cameraDepth', "ViMantic/virtualCameraDepth")
        self.semantic_topic = self.load_param('~topic_result', 'ViMantic/Detections')
        self.cnn_topic = self.load_param('~topic_cnn', 'detectron2_ros/result')
        self.localization_topic = self.load_param('~topic_localization', 'amcl_pose')
        self.image_toCNN = self.load_param('~topic_republic', 'ViMantic/ToCNN')

        # Data parameters
        self.dataset = self.load_param('~dataset', "RobotAtVirtualHome")
        self.data_mode = self.load_param('~data_mode', "real")  # real: with uncertainty / gt: groundtruth
        self.save_colmap = self.load_param('~save_colmap', True)
        self.data_category = self.load_param('~data_category', "mapping") # mapping / localization
        self.save_path = self.load_param('~save_path', "/home/eostajm/datasets/mapirlab/results/")
        self.map_path = self.load_param('~map_path', "/home/josematez/datasets/uhumans2/results/wallmap_no_refined.npy")
        
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
        self.debug = self.load_param('~debug', False)
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
        self._min_area_opening = self.load_param('~min_area_opening', 1000) # Minimum area of an opening to be accepted (in cm2) # 1000
        self._bhattacharyya_threshold = self.load_param('~bhattacharyya_threshold', 10)  # Threshold for the statistical distance of Bhattacharyya  # real 7
        self._euclidean_threshold = self.load_param('~euclidean_threshold', 1.1)  # Threshold for the minimum euclidean distance between walls  # 0.3
        self._eps_alpha = self.load_param('~eps_alpha', 8.0) * pi / 180.0  # Epsilon for DBSCAN of the azimuth angle of the plane (in radians)  # 1
        self._eps_beta = self.load_param('~eps_beta', 8.0) * pi / 180.0  # Epsilon for DBSCAN of the elevation angle of the plane (in radians) # 10
        self._eps_dist = self.load_param('~eps_dist', 0.005)  # Epsilon for DBSCAN of the plane-to-origin distance (in meters)  # real 0.02
        self._num_walls = 0  # Number of current detected walls
        self._walls = {}  # Dictionary including the complete information of the current floorplan

        # Handlers
        self._bridge = CvBridge()
        self._tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self._tfBuffer)
        self._tr = Transformations()
        self._pm = PlaneManager(self._height, self._width, self._cx, self._cy, self._fx, self._fy,
                                self._depth_range_max, self._min_px_opening, self._min_area_opening)
        self._d = Datasets(self.dataset)

        # Transformations
        self._camera_robot_transform = None     # Relative pose of the camera frame w.r.t. the robot frame
        self._map_frame_name = self.load_param('~map_frame_name', "map")

        # Publishers
        self._pub_processed_image = rospy.Publisher(self.image_toCNN, Image, queue_size=1)
        self._walls_pub = rospy.Publisher('sigmafp/walls', MarkerArray, queue_size=10)
        self._pcd_pub = rospy.Publisher('sigmafp/pointcloud_visualization', PointCloud2, queue_size=1)

        # Subscribers
        rospy.Subscriber("wallmap_commands", String, self.callback_commands)

        type_rgb, type_depth = self._d.get_image_msgs_type()
        sub_rgb_image = message_filters.Subscriber(self.image_rgb_topic, type_rgb)
        sub_depth_image = message_filters.Subscriber(self.image_depth_topic, type_depth)
        sub_pose_amcl = message_filters.Subscriber(self.localization_topic, PoseWithCovarianceStamped)

        if self.data_mode == "real":
            rospy.Subscriber(self.cnn_topic, ResultWithWalls, self.callback_new_detection)
            message_filter = message_filters.ApproximateTimeSynchronizer([sub_depth_image, sub_rgb_image, sub_pose_amcl],
                                                                     1, 0.01)
        elif self.data_mode == "gt":
            sub_seg_image = message_filters.Subscriber(self.cnn_topic, Image)
            message_filter = message_filters.ApproximateTimeSynchronizer([sub_depth_image, sub_rgb_image, sub_pose_amcl, sub_seg_image],
                                                                     1, 0.01)

        message_filter.registerCallback(self.callback_synchronize_image)

        

        # Save data for COLMAP
        if self.save_colmap:
            self._ds = DataSaver(self.map_path, self.save_path, self._width, self._height, self._fx, self._fy,
                                self._cx, self._cy)
            
            if self._ds.load_global_map():
                print("[DATASAVER] Global map for wall mask generation has been loaded correctly.")
            else:
                print("[DATASAVER] Loading global map failed. Disabling datasaver.")
                self.save_colmap = False
        
        rospy.logwarn("Initialized")

    ####################################################################################################################
    ################################################### Node Script ####################################################
    ####################################################################################################################

    def run(self):

        rate = rospy.Rate(self._publish_rate)

        while not rospy.is_shutdown():

            # Extracting and projecting detected walls
            if self._flag_processing and self._flag_cnn:

                if self.data_mode == "real":
                    wall_mask = self._bridge.imgmsg_to_cv2(self._last_cnn_result.walls) == 255
                elif self.data_mode == "gt":
                    wall_mask = self._last_msg[-1]

                # Check if there are pixels belonging to walls in the image
                if wall_mask.sum() == 0:
                    self._flag_cnn = False
                    self._flag_processing = False
                    continue
                # Obtain 3D coordinates, in meters, of each pixel
                z = self._last_msg[2]
                x = self._image_c * z
                y = self._image_r * z

                time_start = time.time()

                # Create point cloud of walls
                point_cloud = np.array([z[wall_mask].reshape(-1), x[wall_mask].reshape(-1), y[wall_mask].reshape(-1)]).T

                # Reduce the point cloud to the data in the reliable range of the depth sensor
                point_cloud = point_cloud[np.logical_and(point_cloud[:, 0] > self._min_reliable_cam_depth,
                                                         point_cloud[:, 0] < self._max_reliable_cam_depth)].copy()
            

                # Select randomly indices to downsample point cloud
                rnd_idxs = np.random.default_rng().choice(point_cloud.shape[0], size = min(self._n_points_in_pcd, point_cloud.shape[0]), replace = False)

                # Create the point cloud with the dowmsampled indices
                pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point_cloud[rnd_idxs,:]))

                pcd_np = point_cloud[rnd_idxs,:].copy()

                print("Number of points in downsampled pointcloud: {}/{}".format(pcd_np.shape[0], len(point_cloud)))

                if pcd_np.shape[0] == 0:
                    self._flag_cnn = False
                    self._flag_processing = False
                    continue
                
                # Obtain mean distance between points in the point cloud (VERY TIME-CONSUMING)
                mean_dist_pcd = 0.1
                
                # Compute, normalize and orient normals towards camera location
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=mean_dist_pcd, max_nn=100))
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
                tr_matrix_camera_map = tr_matrix_robot_map @ tr_matrix_robot_camera   # Camera -> Map
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

                # TODO: Testing: trying to include xy coords in the clustering
                dbscan_data1 = np.concatenate((alpha, beta, distance_to_origin, pcd_np[:,0].reshape(-1, 1), pcd_np[:,1].reshape(-1, 1)), axis=1)
                #dbscan_data1 = np.concatenate((alpha, beta, distance_to_origin, np.linalg.norm(pcd_np[:,:2], axis = 1).reshape(-1, 1)), axis=1)
                dbscan_data_scaled1 = dbscan_data1 / np.asarray([self._eps_alpha, self._eps_beta, self._eps_dist, .75, .75])  

                # DBSCAN Clustering
                clustering = DBSCAN(eps=1., min_samples=self._min_points_plane).fit(dbscan_data_scaled1)
                labels = clustering.labels_.astype(np.float_)
                max_label = labels.max()

                #colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
                #colors[labels < 0] = 0
                #pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
                #o3d.visualization.draw_geometries([pcd])

                #plt.scatter(dbscan_data_scaled1[:,0], dbscan_data_scaled1[:,3], c=clustering.labels_.astype(np.float_), cmap="viridis")
                #plt.show()

                # Time for plane extraction
                self._time_plane_extraction += time.time() - time_start

                xyz_img = np.dstack((z,x,y))
                
                if self.save_colmap:
                    
                    # Compute xyz image global
                    pcd_aux = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np.array([z.reshape(-1), x.reshape(-1), y.reshape(-1)]).T))
                    pcd_aux.transform(tr_matrix_camera_map)
                    xyz_img_global = np.array(pcd_aux.points).reshape((xyz_img.shape[0], xyz_img.shape[1], 3))
                    
                    # Set to True if want to add current walls to the reference map for saving data
                    if False:
                        self._ds.add_current_map(deepcopy(self._walls))

                    #planes_mask = self._ds.compute_planes_mask(xyz_img_global, residuals = False, limits = True)
                    planes_mask, planes_mask_ceilingfloor = self._ds.compute_instance_planes_mask(xyz_img_global, residuals = True, limits = True)
                 
                # Characterizing each clustered wall by: its Gaussian distribution and a set of features
                for idx in range(max_label.astype(np.int_) + 1):

                    # Obtaining the set of points belonging to the specific cluster
                    wall_pps = dbscan_data[np.where(labels == idx)[0]]

                    # Computing mean and covariance of the cluster
                    mean_pps = np.mean(wall_pps, axis=0).reshape((3, 1))

                    cov_pps = np.cov(wall_pps.T)

                    # Skipping walls that do not meet the Atlanta world assumption
                    if self.data_mode == "real" and abs((pi / 2.) - mean_pps[1]) > 0.75:
                        continue

                    # Changing the reference system of the Gaussian distribution: from robot to world frame
                    mean_global, cov_global = self._pm.pps_from_robot_to_map(self._last_msg[5], mean_pps, cov_pps)

                    # Too much uncertainty in the robot localization... Skipping data
                    if self.data_mode == "real" and (abs(cov_global[0, 0]) > 0.05 or abs(cov_global[1, 1]) > 0.15):
                        rospy.logwarn("Bad localization, skipping data...")
                        continue

                    # If ground-truth data, no uncertainty from robot localization, just from plane extraction.
                    if self.data_mode == "gt" or self.dataset == "OpenLORIS":
                        cov_global = cov_pps.copy()

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
                    wall_dict["update_unity"] = True

                    plane_width = np.sqrt((wall_dict["max_bound"][0] - wall_dict["min_bound"][0]) ** 2 +
                                          (wall_dict["max_bound"][1] - wall_dict["min_bound"][1]) ** 2)

                    plane_height = wall_dict["max_bound"][2] - wall_dict["min_bound"][2]

                    # Accepting a wall if it has a minimum width... otherwise it is considered as non-informative
                    if plane_width < self._min_plane_width or plane_height < self._min_plane_width:
                        rospy.logwarn("Neglected because not sufficient width.")
                        continue

                    # Extracting openings in walls
                    if False:
                        opening_start = time.time()
                        wall_dict["openings"], plane_info_img = self._pm.detect_openings_in_plane(wall_dict, inv_tr_matrix,
                                                                            self._last_msg[1], np.divide(self._last_msg[2], self._depth_range_max), xyz_img)
                        self._time_opening_detection += time.time() - opening_start
                        self._n_iterations_openings += 1
                    else:
                        wall_dict["openings"] = {}
                        plane_info_img = None
                        self._time_opening_detection += 1
                        self._n_iterations_openings += 1

          
                    self._walls[str(self._num_walls)] = wall_dict
                    self._num_walls += 1


                if self.save_colmap:

                    #planes_mask = np.logical_and(np.logical_and(z > 0.01, z < 5.), planes_mask)

                    #planes_mask = np.logical_and(np.logical_and(y > -1.35, y < 1.5), planes_mask)
                    #planes_mask = np.logical_or(wall_mask.astype(bool), planes_mask)
                    #planes_mask = (255*planes_mask).astype(np.uint8)

                    self._ds.write_data(self._last_msg[1], planes_mask, tr_matrix_camera_map, mode = self.data_category, depth_img = self._last_msg[2], planes_mask_ceilingfloor = planes_mask_ceilingfloor)
                    #self._ds.write_data(self._last_msg[1], planes_mask, tr_matrix_camera_map, mode = "localization")

                    #self._pub_masked_image.publish(self._bridge.cv2_to_imgmsg(cv2.bitwise_and(self._last_msg[1], self._last_msg[1], mask=planes_mask.astype(np.uint8)), 'rgb8'))
                    

                # Another clustering approach
                matching_start = time.time()
                order, plane_features = self._pm.plane_dict_to_features_comp(self._walls)

                # Data association and integration process
                if self.data_mode == "real" and self.dataset != "OpenLORIS" and self.dataset != "TUM":
                    new_walls, num_walls = self._pm.match_and_merge_planes(self._walls, plane_features, order,
                                                                        self._pm.bhattacharyya_distance_features,
                                                                        self._bhattacharyya_threshold,
                                                                        self._euclidean_threshold)
                elif self.data_mode == "gt" or self.dataset == "OpenLORIS" or self.dataset == "TUM":
                    new_walls, num_walls = self._pm.match_and_merge_planes(self._walls, plane_features, order,
                                                                        self._pm.error_free_distance,
                                                                        self._bhattacharyya_threshold,
                                                                        self._euclidean_threshold)

                self._time_plane_matching += time.time() - matching_start

                # Updating the map
                self._num_walls = num_walls
                self._walls = {}
                self._walls = new_walls

                # Showing current floorplan
                self._walls_pub.publish(self._pm.create_msg_walls_markers(self._walls, self._map_frame_name))

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

    #def callback_synchronize_image(self, depth_msg, rgb_msg, pose_msg, seg_msg):
    def callback_synchronize_image(*args):

        if args[0].data_mode == "real":
            self, depth_msg, rgb_msg, pose_msg = args
        elif args[0].data_mode == "gt":
            self, depth_msg, rgb_msg, pose_msg, seg_msg = args

        if not self._flag_processing:
            img_rgb, img_depth = self._d.preprocess_images(rgb_msg, depth_msg, self._depth_range_max)

            if self._camera_robot_transform == None:

                try:
                    self._camera_robot_transform = self._d.get_camera_extrinsics(rgb_msg)
                except:
                    print("Camera robot transform cannot be established.")
            if self.data_mode == "real":
                self._pub_processed_image.publish(self._bridge.cv2_to_imgmsg(img_rgb, 'rgb8'))
                self._tries = 0
                        
                self._last_msg = [rgb_msg.header, img_rgb, img_depth, None, None, pose_msg]
            elif self.data_mode == "gt":
                walls_mask = self._d.preprocess_img_seg(seg_msg)
                self._flag_cnn = True
                self._last_msg = [rgb_msg.header, img_rgb, img_depth, None, None, pose_msg, walls_mask]
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

    def callback_commands(self, data):

        command = data.data

        if not os.path.exists(command) or command[0] != '/':
            command = self.save_path

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

        self._walls = self._pm.global_refinement(deepcopy(self._walls))

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
