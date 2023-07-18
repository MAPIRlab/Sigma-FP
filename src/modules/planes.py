import os
import numpy as np
import rospy
import cv2
import itertools
from copy import deepcopy
from math import pi, sin, cos, atan2
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointField, CompressedImage
import sensor_msgs.point_cloud2 as pc2
from sigmafp.msg import WallMesh, WallMeshArray
import networkx as nx
import time

class PlaneManager(object):

    # Initialization
    def __init__(self, height, width, cx, cy, fx, fy, max_depth, min_px_opening, min_area_opening = 1000):
        """

        :param height: height of the camera image in px
        :param width: width of the camera image in px
        :param cx: horizontal position of the camera center
        :param cy: vertical position of the camera center
        :param fx: focal length in x
        :param fy: focal length in y
        :param max_depth: maximum range of depth sensor
        :param min_px_opening: threshold - minimum number of pixels to accept an opening (just to avoid holes due to
        noise)
        """
        self._height = height
        self._width = width
        self._cx = cx
        self._cy = cy
        self._fx = fx
        self._fy = fy
        self._depth_range_max = max_depth
        self._min_px_opening = min_px_opening
        self._min_area_opening = min_area_opening

    ####################################################################################################################
    ############################################# PPS and Cartesian Spaces #############################################
    ####################################################################################################################

    @staticmethod
    def pps_to_nd(plane):
        """

        :param plane: list with the plane definition in the Plane Parameter Space (alpha, beta, dist_to_origin)
        :return: plane definition following the standard definition of a plane: ax + by + cz = d
        """

        a = cos(plane[0]) * sin(plane[1])
        b = sin(plane[0]) * sin(plane[1])
        c = cos(plane[1])
        d = -plane[2]

        return (a, b, c, d)

    ####################################################################################################################
    ############################################# Advanced Plane Functions #############################################
    ####################################################################################################################

    def plane_intersect(self, plane1, plane2):
        """

        :param plane1: dictionary from the plane1 including the plane representation in the PPS
        :param plane2: dictionary from the plane2 including the plane representation in the PPS
        :return: 2D intersection point of two vertical planes
        """
        plane1 = plane1['mean']
        plane2 = plane2['mean']

        if min((2 * pi) - abs(plane1[0] - plane2[0]), abs(plane1[0] - plane2[0])) < 0.1:
            return None

        a = self.pps_to_nd(plane1)
        b = self.pps_to_nd(plane2)
        a_vec, b_vec = np.array(a[:3], dtype=np.float64), np.array(b[:3], dtype=np.float64)

        aXb_vec = np.cross(a_vec, b_vec)

        A = np.array([a_vec, b_vec, aXb_vec], dtype=np.float64)
        d = np.array([-a[3], -b[3], 0.], dtype=np.float64).reshape(3, 1)

        p_inter = np.linalg.solve(A, d).T[0]

        return np.asarray([p_inter[0], p_inter[1], 0.], dtype=np.float64)

    @staticmethod
    def min_distance_to_intersection(wall_corners, pt):
        """

        :param wall_corners: list with the corners of a wall
        :param pt: intersection point
        :return: minimum distance from the wall to the intersection corner
        """

        dist1 = np.linalg.norm(wall_corners[0][:2] - pt[:2])
        dist2 = np.linalg.norm(wall_corners[-1][:2] - pt[:2])

        angle1 = atan2(pt[1] - wall_corners[0][1], pt[0] - wall_corners[0][0])
        angle2 = atan2(pt[1] - wall_corners[1][1], pt[0] - wall_corners[1][0])
        inside = True
        if min((2 * pi) - abs(angle1 - angle2), abs(angle1 - angle2)) < 0.05:
            inside = False

        if dist1 < dist2:
            return dist1, wall_corners[0], inside

        else:
            return dist2, wall_corners[-1], inside


    def match_and_merge_planes_old(self, walls_dict, plane_features, order, clustering_dist,
                               clustering_th, euclidean_th=0.1):
        """

        :param walls_dict: dictionary representing the 3D floorplan
        :param plane_features: features obtained from the 3D floorplan dictionary
        :param order: reference order between the original walls_dict and the plane_features matrix
        :param clustering_dist: name of the function of the statistical distance (note that you can implement your
        own function)
        :param clustering_th: minimum Bhattacharya distance to consider a match
        :param euclidean_th: minimum Euclidean distance to consider a match
        :return: updated dictionary representing the 3D floorplan
        """

        new_walls = {}
        num_walls = 0

        if len(plane_features) > 0:
            
            clustering = DBSCAN(eps=clustering_th, min_samples=1, metric=clustering_dist).fit(plane_features)
            labels = clustering.labels_.astype(np.float_)
            max_label = labels.max()

            for idx in range(max_label.astype(np.int_) + 1):

                keys = np.where(labels == idx)[0]            
                keys = [order[key] for key in keys]
                planes = [[id_key, walls_dict[key]] for id_key, key in enumerate(keys)]

                if len(planes) > 1:
                    planes_combinations = list(itertools.combinations(planes, 2))
                    fuse_ok = len(planes) * [False]
                    for pack_planes in planes_combinations:
                        corners_plane1 = np.asarray(self.get_plane_corners(pack_planes[0][1])).reshape((-1, 3))[:,
                                         :2]
                        corners_plane2 = np.asarray(self.get_plane_corners(pack_planes[1][1])).reshape((-1, 3))[:,
                                         :2]
                        min_euc_dist = np.abs(distance.cdist(corners_plane1, corners_plane2)).min()
                        max_bound = pack_planes[0][1]["max_bound"]
                        min_bound = pack_planes[0][1]["min_bound"]
                        half_plane_width = 0.5 * np.sqrt((max_bound[0] - min_bound[0]) ** 2 +
                                                         (max_bound[1] - min_bound[1]) ** 2)
                        plane_center = np.asarray([[0.5 * (max_bound[0] + min_bound[0]),
                                                    0.5 * (max_bound[1] + min_bound[1])]])

                        dist_corner_center = np.abs(distance.cdist(corners_plane2, plane_center)).min()

                        if (pack_planes[0][1]["max_bound"][0] < -2. and pack_planes[0][1]["min_bound"][0] > -6.5) and (pack_planes[1][1]["max_bound"][0] < -2. and pack_planes[1][1]["min_bound"][0] > -6.5):
                            print("AQUI")
                            print("Plane 1: ")
                            print("max bound: " + str(pack_planes[0][1]["max_bound"]))
                            print("min bound: " + str(pack_planes[0][1]["min_bound"]))
                            print("mean: " + str(pack_planes[0][1]["mean"]))
                            print("Corners: " + str(corners_plane1))
                            print("Plane 2: ")
                            print("max bound: " + str(pack_planes[1][1]["max_bound"]))
                            print("min bound: " + str(pack_planes[1][1]["min_bound"]))
                            print("mean: " + str(pack_planes[1][1]["mean"]))
                            print("Corners: " + str(corners_plane2))
                            print("Distance corner center: " + str(dist_corner_center))
                            print("Min euc dist: " + str(min_euc_dist))
                            print(min_euc_dist < euclidean_th, dist_corner_center < half_plane_width)

                        if min_euc_dist < euclidean_th or dist_corner_center < half_plane_width:

                            fuse_ok[pack_planes[0][0]] = True
                            fuse_ok[pack_planes[1][0]] = True


                    planes_to_fuse = [plane[1] for plane in planes if fuse_ok[plane[0]]]

                    if len(planes_to_fuse) > 0:
                        print("Fusing " + str(len(planes_to_fuse)) + "planes... ")
                        for i in planes_to_fuse:
                            print(i["mean"])
                        print("-")    
                        new_walls[str(num_walls)] = self.fuse_n_planes_prob(planes_to_fuse)
                        num_walls += 1

                    planes_to_not_fuse = [plane[1] for plane in planes if not (fuse_ok[plane[0]])]
                    for plane in planes_to_not_fuse:
                        print("Adding matched planes but no distance meet... " + str(plane["mean"]))
                        new_walls[str(num_walls)] = plane
                        num_walls += 1

                else:
                    print("Adding no matched plane... " + str(planes[0][1]["mean"]))
                    new_walls[str(num_walls)] = planes[0][1]
                    num_walls += 1

        return new_walls, num_walls

    def match_and_merge_planes(self, walls_dict, plane_features, order, clustering_dist,
                                   clustering_th, euclidean_th=0.1):
        """

        :param walls_dict: dictionary representing the 3D floorplan
        :param plane_features: features obtained from the 3D floorplan dictionary
        :param order: reference order between the original walls_dict and the plane_features matrix
        :param clustering_dist: name of the function of the statistical distance (note that you can implement your
        own function)
        :param clustering_th: minimum Bhattacharya distance to consider a match
        :param euclidean_th: minimum Euclidean distance to consider a match
        :return: updated dictionary representing the 3D floorplan
        """

        new_walls = {}
        num_walls = 0

        if len(plane_features) > 0:
            
            #clustering = DBSCAN(eps=clustering_th, min_samples=1, metric=clustering_dist).fit(plane_features)

            labels = self.histogram_based_clustering(plane_features[:,:3])
            max_label = labels.max()

            wall_graph = nx.Graph()
            wall_graph.add_nodes_from(walls_dict.keys())
            

            for idx in range(max_label.astype(np.int_) + 1):

                keys = np.where(labels == idx)[0]            
                keys = [order[key] for key in keys]

                if len(keys) > 1:
                    planes_combinations = list(itertools.combinations(keys, 2))

                    for k1, k2 in planes_combinations:
                        
                        if clustering_dist(plane_features[order.index(k1),:], plane_features[order.index(k2),:]) > clustering_th:
                            continue

                        corners_plane1 = np.asarray(self.get_plane_corners(walls_dict[str(k1)])).reshape((-1, 3))[:,:2]
                        corners_plane2 = np.asarray(self.get_plane_corners(walls_dict[str(k2)])).reshape((-1, 3))[:,:2]

                        min_euc_dist = np.abs(distance.cdist(corners_plane1, corners_plane2)).min()

                        max_bound = walls_dict[str(k1)]["max_bound"][:2]
                        min_bound = walls_dict[str(k1)]["min_bound"][:2]


                        half_plane_width = 0.5 * np.linalg.norm(max_bound - min_bound)
                        plane_center = (max_bound + min_bound) / 2.0

                        dist_corner_center = np.abs(distance.cdist(corners_plane2, plane_center.reshape((-1,2)))).min()

                        if min_euc_dist < euclidean_th or dist_corner_center < half_plane_width:

                            wall_graph.add_edge(str(k1), str(k2))



            matched_subgraphs = [list(c) for c in nx.connected_components(wall_graph)]

            for sg in matched_subgraphs:

                if len(sg) > 1:
                    planes_to_fuse = [walls_dict[i] for i in sg]
                    new_walls[str(num_walls)] = self.fuse_n_planes_prob(planes_to_fuse)
                    num_walls += 1
                else:
                    new_walls[str(num_walls)] = walls_dict[sg[0]]
                    num_walls += 1
                    

        return new_walls, num_walls

    @staticmethod
    def histogram_based_clustering(data):
        
        labels = np.zeros((data.shape[0]), dtype = int)
        
        data[:,0] = (data[:,0]+np.pi) % (2*np.pi) - np.pi
        x_bins = np.linspace(-np.pi, np.pi, num=12)
        #x_bins = np.histogram_bin_edges(data[:,0], bins="auto")
        y_bins = np.histogram_bin_edges(data[:,2], bins="auto")

        votes, ranges_x, ranges_y = np.histogram2d(data[:,0], data[:,2], bins = [x_bins, y_bins])

        votes[0,:] += votes[-1,:]
        votes = votes[:-1,:]

        clusters = np.where(votes > 0)

        label = 0
        for i,j in zip(*clusters):
            x_valids = np.logical_and(ranges_x[i] <= data[:,0], data[:,0] < ranges_x[i+1])
            y_valids = np.logical_and(ranges_y[j] <= data[:,2], data[:,2] <= ranges_y[j+1])
            valids = np.logical_and(x_valids, y_valids)
            labels[valids] = label

            if i == 0:
                x_valids = np.logical_and(ranges_x[-2] <= data[:,0], data[:,0] < ranges_x[-1])
                valids = np.logical_and(x_valids, y_valids)
                labels[valids] = label

            label += 1

        return labels


    def detect_openings_in_plane(self, plane, transform, img_rgb, img_depth):
        """

        :param plane: dictionary including the plane features
        :param transform: camera-world transform
        :param img_rgb: RGB image just for visualization
        :param img_depth: depth image to detect openings
        :return: openings (if they have been detected)
        """
        corners = self.get_plane_corners(plane)

        wall_width = np.linalg.norm(np.array(plane["max_bound"])[:2] - np.array(plane["min_bound"])[:2])

        wall_height = plane["max_bound"][2] - plane["min_bound"][2]

        corners_in_camera_frame = [np.matmul(transform, np.concatenate((corner, [1.])).reshape(-1, 1)) for corner in
                                   corners]

        corners_in_camera_frame = [corner[:3] / corner[3] for corner in corners_in_camera_frame]
        
        corners_with_depth = [[int(self._cx - (pt[1] * self._fx / pt[0])),
                               int(self._cy - (pt[2] * self._fy / pt[0])),
                               pt[0]] for pt in corners_in_camera_frame]
                               

        corners_in_px = [corner[:2] for corner in corners_with_depth]
        
        img_show = img_rgb.copy()

        mask = np.zeros((self._height, self._width), dtype=np.uint8)

        cv2.fillPoly(mask, [np.asarray(corners_in_px).reshape((-1, 1, 2))], (255))

        img_d = cv2.bitwise_and(img_depth, img_depth, mask=mask.astype(np.uint8))

        pixel_coords_top = sorted(sorted(corners_in_px, key=lambda x: x[0], reverse=True)[:2], key=lambda x: x[1],
                                  reverse=True)
        pixel_coords_bottom = sorted(sorted(corners_in_px, key=lambda x: x[0], reverse=True)[2:], key=lambda x: x[1],
                                     reverse=True)
        pixel_coords = np.concatenate((pixel_coords_top, pixel_coords_bottom))


        real_world_list = np.asarray([(round(100. * wall_width) - 1, round(100. * wall_height) - 1),
                                      (round(100. * wall_width) - 1, 0.),
                                      (0., 0. + round(100. * wall_height) - 1),
                                      (0., 0.)])
        
        #if abs(ar_px - ar_r) > 0.1:
        #    print(ar_px, ar_r)

        #M = cv2.findHomography(pixel_coords, real_world_list)[0]

        M = cv2.getPerspectiveTransform(pixel_coords.astype(np.float32), real_world_list.astype(np.float32))

        try:
            inv_M = np.linalg.inv(M)
            corners_non_perspective = [[np.matmul(M, np.concatenate((corner[:2], [1])).reshape(-1, 1)), corner[-1]]
                                       for corner in corners_with_depth]
            
            corners_good = [[int((corner[0][0] / corner[0][2])[0]), int((corner[0][1] / corner[0][2])[0]), corner[-1]]
                            for corner in corners_non_perspective]


            px_max_x = max(corners_good, key=lambda x: x[0])
            px_min_x = min(corners_good, key=lambda x: x[0])

            depth_min_x = float(np.clip(px_min_x[-1], 0.0, self._depth_range_max) / self._depth_range_max)
            depth_max_x = float(np.clip(px_max_x[-1], 0.0, self._depth_range_max) / self._depth_range_max)

            mask_h = np.tile(np.linspace(depth_min_x, depth_max_x, int(px_max_x[0]) - int(px_min_x[0])),
                             (2*self._height, 1))


        except:
            rospy.logwarn("Error computing inverse homography...")
            px_min_x = None
            px_max_x = None
            mask_h = None
            return {}, None

        mask_h_big = np.zeros((2*self._height, 2*self._width), dtype=np.float64)

        try:
            mask_h_big[:, int(px_min_x[0]):int(px_max_x[0])] = mask_h
        except:
            rospy.logwarn("Error mask estimated depth image.")
            return {}, None

        

        mask_h_big = cv2.warpPerspective(mask_h_big, inv_M, (self._width, self._height))
        mask_h_big = cv2.bitwise_and(mask_h_big, mask_h_big, mask=mask.astype(np.uint8))

        diff = img_d - mask_h_big

        mask_doors = np.zeros((self._height, self._width), dtype=np.uint8)
        mask_doors[np.logical_and(diff > 0.05, mask_h_big > 0)] = 255

        mask_doors = cv2.erode(mask_doors, np.ones((7, 7), np.uint8))
        mask_doors = cv2.dilate(mask_doors, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(mask_doors, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        #openings = [opening for opening in contours if cv2.contourArea(opening) > self._min_px_opening]
        openings = [opening for opening in contours if cv2.contourArea(opening) > 2000]

        camera_origin = np.asarray([0., 0., 0.], dtype=np.float64)
        camera_origin_world = np.matmul(np.linalg.inv(transform),
                                        np.concatenate((camera_origin, [1])).reshape((-1, 1)))
        camera_origin_world = camera_origin_world[:3] / camera_origin_world[3]

        openings_dict = {}
        idx = 0
        for opening in openings:
            hull = cv2.convexHull(opening)
            corners = cv2.approxPolyDP(hull, 0.025 * cv2.arcLength(opening, True), True)
            if len(corners) > 4:
                coef = 0.02
                while len(corners) > 4 and coef > 0.0:
                    corners = cv2.approxPolyDP(hull, coef * cv2.arcLength(opening, True), True)
                    coef -= 0.05
            if len(corners) == 4:
                openings_center_px = np.mean(opening, axis = 0)[0].astype(np.uint)
                img_show = cv2.circle(img_show, (openings_center_px[0], openings_center_px[1]), 5, (0, 255, 0), -1)         

                projections_3D_camera = []
                for corner in corners:
                    img_show = cv2.circle(img_show, (corner[0][0], corner[0][1]), 5, (0, 255, 0), -1)
                    projections_3D_camera.append([[1.0, (self._cx - corner[0][0]) / float(self._fx),
                                                   (self._cy - corner[0][1]) / float(self._fy)] for corner in corners])

                projections_3D_camera = projections_3D_camera[0]
                projections_3D_world = [
                    np.matmul(np.linalg.inv(transform), np.concatenate((line, [1])).reshape((-1, 1)))
                    for line in projections_3D_camera]

                alpha = plane["mean"][0]
                beta = plane["mean"][1]
                distance_origin = plane["mean"][2]
                plane_equation = np.asarray([sin(beta) * cos(alpha), sin(beta) * sin(alpha), cos(beta),
                                             -distance_origin])

                corners_3D_world = [np.asarray(self.isect_line_plane_v3_4d(camera_origin_world, point[:3],
                                                                           plane_equation)).tolist()
                                    for point in projections_3D_world]

                opening_max_bound = [max(map(lambda x: x[0], corners_3D_world))[0],
                                    max(map(lambda x: x[1], corners_3D_world))[0],
                                    max(map(lambda x: x[2], corners_3D_world))[0]]
                
                opening_min_bound = [min(map(lambda x: x[0], corners_3D_world))[0],
                                    min(map(lambda x: x[1], corners_3D_world))[0],
                                    min(map(lambda x: x[2], corners_3D_world))[0]]

                opening_width = 100.* np.linalg.norm(np.array(opening_max_bound)[:2] - np.array(opening_min_bound)[:2])
                opening_height = 100. * np.array(opening_max_bound)[2] - np.array(opening_min_bound)[2]

                if opening_width * opening_height >= self._min_area_opening:
                    openings_dict[str(idx)] = {}

                    openings_dict[str(idx)]["max_bound"] = opening_max_bound

                    openings_dict[str(idx)]["min_bound"] = opening_min_bound

                    openings_dict[str(idx)]["center"] = np.asarray([0.5 * (openings_dict[str(idx)]["max_bound"][0] +
                                                                        openings_dict[str(idx)]["min_bound"][0]),
                                                                    0.5 * (openings_dict[str(idx)]["max_bound"][1] +
                                                                        openings_dict[str(idx)]["min_bound"][1])])

                    idx += 1

        #cv2.imshow("Openings", img_show)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        return openings_dict, [M, pixel_coords, real_world_list.astype(int)]

    def plane_dict_to_features_comp(self, planes_dict):
        """

        :param planes_dict: dictionary with the 3D floorplan
        :return: array representing the features matrix of the current 3D floorplan and an order matrix to relate the
        dictionary with the features matrix.
        """
        order = []
        plane_features = np.empty((len(planes_dict), 20), dtype=np.float64)

        for idx, plane_key in enumerate(planes_dict):
            order.append(plane_key)
            plane_corners = np.asarray(self.get_plane_corners(planes_dict[plane_key]))
            plane_features[idx, :] = np.concatenate((planes_dict[plane_key]["mean"].reshape(-1),
                                                     planes_dict[plane_key]["cov"].reshape(-1),
                                                     plane_corners[:, :2].reshape(-1)), axis=0)

        return order, plane_features

    
    @staticmethod
    def get_plane_corners(plane):
        """ THIS FUNCTION IS OUT-OF-DATE. NOW, WE USE get_plane_corners_with_full_openings_triangulation FUNCTION"""

        max_bound = plane["max_bound"]
        min_bound = plane["min_bound"]

        half_plane_width = 0.5 * np.sqrt((max_bound[0] - min_bound[0]) ** 2 + (max_bound[1] - min_bound[1]) ** 2)
        plane_center = np.asarray([0.5 * (max_bound[0] + min_bound[0]), 0.5 * (max_bound[1] + min_bound[1])])

        angle = -((pi / 2.0) - plane["mean"][0])
        point_right = np.asarray([plane_center[0] + half_plane_width * cos(angle),
                                  plane_center[1] + half_plane_width * sin(angle)])
        point_left = np.asarray([plane_center[0] - half_plane_width * cos(angle),
                                 plane_center[1] - half_plane_width * sin(angle)])

        points = [np.concatenate((point_right, np.asarray([min_bound[2]]))),
                  np.concatenate((point_right, np.asarray([max_bound[2]]))),
                  np.concatenate((point_left, np.asarray([max_bound[2]]))),
                  np.concatenate((point_left, np.asarray([min_bound[2]])))]

        corrected_points = []

        alpha = plane["mean"][0]
        beta = plane["mean"][1]
        distance_origin = plane["mean"][2]
        normal_plane = np.asarray([sin(beta) * cos(alpha), sin(beta) * sin(alpha), cos(beta)])

        for point in points:
            corrected_points.append(
                point + (distance_origin - sum(point * normal_plane)) / sum(normal_plane * normal_plane)
                * normal_plane)

        return corrected_points
    


    def compute_plane_dimensions(self, plane):

        points = self.get_plane_corners(plane)
        # return width, height
        return np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[0])

    @staticmethod
    def get_plane_corners_with_full_openings_triangulation(plane):
        """

        :param plane: plane dictionary including features
        :return: a list with a set of triangles representing the whole wall including openings
        """

        max_bound = plane["max_bound"]
        min_bound = plane["min_bound"]

        half_plane_width = 0.5 * np.sqrt((max_bound[0] - min_bound[0]) ** 2 + (max_bound[1] - min_bound[1]) ** 2)
        plane_center = np.asarray([0.5 * (max_bound[0] + min_bound[0]), 0.5 * (max_bound[1] + min_bound[1])])

        angle = -((pi / 2.0) - plane["mean"][0])

        alpha = plane["mean"][0]
        beta = plane["mean"][1]
        distance_origin = plane["mean"][2]
        normal_plane = np.asarray([sin(beta) * cos(alpha), sin(beta) * sin(alpha), cos(beta)])

        point_right = np.asarray([plane_center[0] + half_plane_width * cos(angle),
                                  plane_center[1] + half_plane_width * sin(angle)])
        point_left = np.asarray([plane_center[0] - half_plane_width * cos(angle),
                                 plane_center[1] - half_plane_width * sin(angle)])

        points = [np.concatenate((point_left, np.asarray([min_bound[2]]))),
                  np.concatenate((point_left, np.asarray([max_bound[2]])))]

        openings_class = []

        if len(plane["openings"].keys()) > 0:
            distances = [[key, np.linalg.norm(plane["openings"][key]["center"] - point_left)]
                         for key in plane["openings"].keys()]

            ordered_distances = sorted(distances, key=lambda x: x[1])

            for opening in ordered_distances:
                opening_max_bound = plane["openings"][str(opening[0])]["max_bound"]
                opening_min_bound = plane["openings"][str(opening[0])]["min_bound"]
                opening_center = plane["openings"][str(opening[0])]["center"]

                half_opening_width = 0.5 * np.sqrt((opening_max_bound[0] - opening_min_bound[0]) ** 2 +
                                                   (opening_max_bound[1] - opening_min_bound[1]) ** 2)

                point_opening_right = np.asarray([opening_center[0] + half_opening_width * cos(angle),
                                                  opening_center[1] + half_opening_width * sin(angle)])
                point_opening_left = np.asarray([opening_center[0] - half_opening_width * cos(angle),
                                                 opening_center[1] - half_opening_width * sin(angle)])

                if opening_min_bound[2] < 0.6:  # Its a door!
                    openings_class.append('door')
                    points.append(np.concatenate((point_opening_left, np.asarray([min_bound[2]]))))
                    points.append(np.concatenate((point_opening_left, np.asarray([opening_max_bound[2]]))))
                    points.append(np.concatenate((point_opening_right, np.asarray([opening_max_bound[2]]))))
                    points.append(np.concatenate((point_opening_right, np.asarray([min_bound[2]]))))

                else:  # Its a window!
                    openings_class.append('window')
                    points.append(np.concatenate((point_opening_left, np.asarray([opening_min_bound[2]]))))
                    points.append(np.concatenate((point_opening_left, np.asarray([opening_max_bound[2]]))))
                    points.append(np.concatenate((point_opening_right, np.asarray([opening_max_bound[2]]))))
                    points.append(np.concatenate((point_opening_right, np.asarray([opening_min_bound[2]]))))

        points.append(np.concatenate((point_right, np.asarray([max_bound[2]]))))
        points.append(np.concatenate((point_right, np.asarray([min_bound[2]]))))

        triangles_vertex = []

        n_polygons = 1 + 2 * len(openings_class)

        if n_polygons == 1:
            top_vertices = sorted(points[0:4], key=lambda x: x[-1], reverse=True)[:2]
            top_vertices_ordered = sorted(top_vertices,
                                          key=lambda x: atan2((x[0] - plane_center[0]) * normal_plane[1] -
                                                              (x[1] - plane_center[1]) * normal_plane[0],
                                                              (x[0] - plane_center[0]) * normal_plane[0] +
                                                              (x[1] - plane_center[1]) * normal_plane[1]),
                                          reverse=True)
            bot_vertices = [np.asarray([x[0], x[1], min_bound[2]]) for x in top_vertices_ordered]
            triangles_vertex.append(bot_vertices[0])
            triangles_vertex += top_vertices_ordered
            triangles_vertex.append(top_vertices_ordered[1])
            triangles_vertex += bot_vertices[::-1]
            # triangles_vertex += points[0:3]
            # triangles_vertex.append(points[0])
            # triangles_vertex += points[2:]
        else:
            for p_idx in range(n_polygons):
                top_vertices = sorted(points[2 * p_idx:2 * p_idx + 4], key=lambda x: x[-1], reverse=True)[:2]
                bot_vertices = sorted(points[2 * p_idx:2 * p_idx + 4], key=lambda x: x[-1], reverse=True)[2:]
                triangle_center = np.mean(np.asarray(top_vertices), axis=0)
                top_vertices_ordered = sorted(top_vertices,
                                              key=lambda x: atan2((x[0] - triangle_center[0]) * normal_plane[1] -
                                                                  (x[1] - triangle_center[1]) * normal_plane[0],
                                                                  (x[0] - triangle_center[0]) * normal_plane[0] +
                                                                  (x[1] - triangle_center[1]) * normal_plane[1]),
                                              reverse=True)
                if p_idx % 2 != 0:
                    if openings_class[((p_idx + 1) // 2) - 1] == 'door':
                        z_door = top_vertices[-1][-1]
                        top_vertices_ordered = [np.asarray([x[0], x[1], max_bound[2]]) for x in top_vertices_ordered]
                        bot_vertices = [np.asarray([x[0], x[1], z_door]) for x in top_vertices_ordered]
                        triangles_vertex.append(bot_vertices[0])
                        triangles_vertex += top_vertices_ordered
                        triangles_vertex.append(top_vertices_ordered[1])
                        triangles_vertex += bot_vertices[::-1]
                    else:
                        z_max_window = top_vertices[-1][-1]
                        z_min_window = bot_vertices[0][-1]
                        top_vertices_1 = [np.asarray([x[0], x[1], max_bound[2]]) for x in top_vertices_ordered]
                        bot_vertices_1 = [np.asarray([x[0], x[1], z_max_window]) for x in top_vertices_1]
                        triangles_vertex.append(bot_vertices_1[0])
                        triangles_vertex += top_vertices_1
                        triangles_vertex.append(top_vertices_1[1])
                        triangles_vertex += bot_vertices_1[::-1]
                        top_vertices_2 = [np.asarray([x[0], x[1], z_min_window]) for x in top_vertices_ordered]
                        bot_vertices_2 = [np.asarray([x[0], x[1], min_bound[2]]) for x in top_vertices_2]
                        triangles_vertex.append(bot_vertices_2[0])
                        triangles_vertex += top_vertices_2
                        triangles_vertex.append(top_vertices_2[1])
                        triangles_vertex += bot_vertices_2[::-1]
                        # top_vertices_2 = [np.asarray([x[0], x[1], z_min_window]) for x in bot_vertices]
                        # bot_vertices_2 = [np.asarray([x[0], x[1], min_bound[2]]) for x in top_vertices_2]
                        # triangles_vertex += top_vertices_2
                        # triangles_vertex.append(bot_vertices_2[0])
                        # triangles_vertex += bot_vertices_2
                        # triangles_vertex.append(top_vertices_2[1])
                else:
                    top_vertices_ordered = [np.asarray([x[0], x[1], max_bound[2]]) for x in top_vertices_ordered]
                    bot_vertices = [np.asarray([x[0], x[1], min_bound[2]]) for x in top_vertices_ordered]
                    triangles_vertex.append(bot_vertices[0])
                    triangles_vertex += top_vertices_ordered
                    triangles_vertex.append(top_vertices_ordered[1])
                    triangles_vertex += bot_vertices[::-1]

        corrected_points = []

        for point in triangles_vertex:
            corrected_points.append(
                point + (distance_origin - sum(point * normal_plane)) / sum(normal_plane * normal_plane)
                * normal_plane)

        return corrected_points

    def fuse_n_planes_prob(self, planes, two_sided=False):
        """

        :param planes: a list of planes to fuse
        :param two_sided: boolean to fuse both sides of a wall, avoiding to represent wall thickness - not tested (not
        recommended)
        :return: dictionary of the fused plane
        """
        if len(planes) == 0:
            return None

        mean_sin_1 = 0
        mean_cos_1 = 0
        mean_sin_2 = 0
        mean_cos_2 = 0
        sum_distance = 0

        main_plane = min(planes, key=lambda x: x['first_seen'])

        fused_plane = {}
        fused_plane["max_bound"] = planes[0]["max_bound"]
        fused_plane["min_bound"] = planes[0]["min_bound"]
        fused_plane["cov"] = np.zeros((3, 3), dtype=np.float64)
        fused_plane["n_samples"] = sum([plane["n_samples"] for plane in planes])
        fused_plane["n_detections"] = sum([plane["n_detections"] for plane in planes])
        fused_plane["openings"] = {}
        fused_plane["color"] = main_plane["color"]
        fused_plane["first_seen"] = main_plane["first_seen"]
        

        # Note, this try-except is due to the version without Unity visualization does not contain this feature.
        try:
            fused_plane["update_unity"] = True
        except:
            pass

        for idx in range(len(planes)):
            fused_plane["max_bound"] = np.max(np.vstack((fused_plane["max_bound"], planes[idx]["max_bound"])), axis=0)
            fused_plane["min_bound"] = np.min(np.vstack((fused_plane["min_bound"], planes[idx]["min_bound"])), axis=0)

            if two_sided:
                diff_angles = abs(planes[idx]["mean"][0] - main_plane['mean']['0'])
                if cos(diff_angles) < -0.8:
                    planes[idx]["mean"][0] -= pi
                    planes[idx]["mean"][2] = -planes[idx]["mean"][2]

            mean_sin_1 += planes[idx]["n_samples"] * sin(planes[idx]["mean"][0])
            mean_cos_1 += planes[idx]["n_samples"] * cos(planes[idx]["mean"][0])
            mean_sin_2 += planes[idx]["n_samples"] * sin(planes[idx]["mean"][1])
            mean_cos_2 += planes[idx]["n_samples"] * cos(planes[idx]["mean"][1])
            sum_distance += planes[idx]["n_samples"] * planes[idx]["mean"][2]
            fused_plane["cov"] += (float(planes[idx]["n_samples"]) ** 2 / float(fused_plane["n_samples"]) ** 2) \
                                  * planes[idx]["cov"]

        fused_plane["mean"] = np.array([[atan2(mean_sin_1 / fused_plane["n_samples"],
                                               mean_cos_1 / fused_plane["n_samples"])],
                                        [atan2(mean_sin_2 / fused_plane["n_samples"],
                                               mean_cos_2 / fused_plane["n_samples"])],
                                        [sum_distance / fused_plane["n_samples"]]], dtype=np.float64)
        
        fused_plane["openings"] =self.fuse_openings(fused_plane, planes)


        return fused_plane

    @staticmethod
    def fuse_openings_complete(fused_plane, planes):
        
        fused_openings = {}

        reference_point = fused_plane["min_bound"].copy()
        reference_point[2] = fused_plane["max_bound"][2]

        plane_width = np.linalg.norm(fused_plane["max_bound"][:2] - fused_plane["min_bound"][:2])
        plane_height = fused_plane["max_bound"][2] - fused_plane["min_bound"][2]

        img_openings = np.ones((int(100. * plane_height), int(100. * plane_width)), dtype = np.int) * -1

        # Fusing openings
        openings = []
        order = []

        label = 0

        for idx, plane in enumerate(planes):
            for opening_key in plane["openings"].keys():

                max_uv = [min(int(100. * (np.linalg.norm(plane["openings"][opening_key]["max_bound"][:2] - reference_point[:2]) / plane_width)), int(100. * plane_width)),
                          min(int(100. * ((reference_point[2] - plane["openings"][opening_key]["max_bound"][2]) / plane_height)), int(100. * plane_height))]

                min_uv = [min(int(100. * (np.linalg.norm(plane["openings"][opening_key]["min_bound"][:2] - reference_point[:2]) / plane_width)), int(100. * plane_width)),
                          min(int(100. * ((reference_point[2] - plane["openings"][opening_key]["min_bound"][2]) / plane_height)), int(100. * plane_height))]

                if img_openings[max_uv[1]:min_uv[1], min_uv[0]:max_uv[0]].max() == -1:
                    
                    img_openings[max_uv[1]:min_uv[1], min_uv[0]:max_uv[0]] = label
                    fused_openings[str(label)] = {}
                    fused_openings[str(label)]["min_bound"] = plane["openings"][opening_key]["min_bound"].copy()
                    fused_openings[str(label)]["max_bound"] = plane["openings"][opening_key]["max_bound"].copy()

                    label += 1

                else:

                    opening_idx = img_openings[max_uv[1]:min_uv[1], min_uv[0]:max_uv[0]].max()

                    img_openings[max_uv[1]:min_uv[1], min_uv[0]:max_uv[0]] = opening_idx

                    fused_openings[str(opening_idx)]["min_bound"] = np.min(np.vstack((fused_openings[str(opening_idx)]["min_bound"], plane["openings"][opening_key]["min_bound"])), axis=0)
                    fused_openings[str(opening_idx)]["max_bound"] = np.max(np.vstack((fused_openings[str(opening_idx)]["max_bound"], plane["openings"][opening_key]["max_bound"])), axis=0)

        for opening_key in fused_openings.keys():

            fused_openings[str(opening_key)]["center"] = np.asarray([0.5 * (fused_openings[str(opening_key)]["max_bound"][0] + fused_openings[str(opening_key)]["min_bound"][0]),
                                                                      0.5 * (fused_openings[str(opening_key)]["max_bound"][1] + fused_openings[str(opening_key)]["min_bound"][1])])
            
            if fused_openings[str(opening_key)]["min_bound"][2] < 1.15:
                fused_openings[str(opening_key)]["min_bound"][2] = fused_plane["min_bound"][2]


        return fused_openings

    @staticmethod
    def fuse_openings(fused_plane, planes):
        
        fused_openings = {}

        reference_point = np.array(fused_plane["min_bound"]).copy()
        reference_point[2] = fused_plane["max_bound"][2]

        plane_width = np.linalg.norm(fused_plane["max_bound"][:2] - fused_plane["min_bound"][:2])
        plane_height = fused_plane["max_bound"][2] - fused_plane["min_bound"][2]

        img_openings = np.ones((int(100. * plane_height), int(100. * plane_width)), dtype = np.int) * -1

        # Fusing openings
        openings = []
        order = []

        label = 0

        for idx, plane in enumerate(planes):
            for opening_key in plane["openings"].keys():

                max_uv = [min(int(100. * plane_width * (np.linalg.norm(plane["openings"][opening_key]["max_bound"][:2] - reference_point[:2]) / plane_width)), int(100. * plane_width)),
                          min(int(100. * plane_height * ((reference_point[2] - plane["openings"][opening_key]["max_bound"][2]) / plane_height)), int(100. * plane_height))]

                min_uv = [min(int(100. * plane_width * (np.linalg.norm(plane["openings"][opening_key]["min_bound"][:2] - reference_point[:2]) / plane_width)), int(100. * plane_width)),
                          min(int(100. * plane_height * ((reference_point[2] - plane["openings"][opening_key]["min_bound"][2]) / plane_height)), int(100. * plane_height))]

                if img_openings[:, min_uv[0]:max_uv[0]].max() == -1:
                    
                    img_openings[:, min_uv[0]:max_uv[0]] = label
                    fused_openings[str(label)] = {}
                    fused_openings[str(label)]["min_bound"] = plane["openings"][opening_key]["min_bound"].copy()
                    fused_openings[str(label)]["max_bound"] = plane["openings"][opening_key]["max_bound"].copy()

                    label += 1

                else:

                    opening_idx = img_openings[:, min_uv[0]:max_uv[0]].max()

                    img_openings[:, min_uv[0]:max_uv[0]] = opening_idx

                    fused_openings[str(opening_idx)]["min_bound"] = np.min(np.vstack((fused_openings[str(opening_idx)]["min_bound"], plane["openings"][opening_key]["min_bound"])), axis=0)
                    fused_openings[str(opening_idx)]["max_bound"] = np.max(np.vstack((fused_openings[str(opening_idx)]["max_bound"], plane["openings"][opening_key]["max_bound"])), axis=0)

        for opening_key in fused_openings.keys():

            fused_openings[str(opening_key)]["center"] = np.asarray([0.5 * (fused_openings[str(opening_key)]["max_bound"][0] + fused_openings[str(opening_key)]["min_bound"][0]),
                                                                      0.5 * (fused_openings[str(opening_key)]["max_bound"][1] + fused_openings[str(opening_key)]["min_bound"][1])])

            #if fused_openings[str(opening_key)]["min_bound"][2] < 1.2:
            #    fused_openings[str(opening_key)]["min_bound"][2] = fused_plane["min_bound"][2]

        return fused_openings

                

    @staticmethod
    def fuse_openings_old(fused_plane, planes):
        
        fused_openings = {}
        # Fusing openings
        openings = []
        order = []
        for idx, plane in enumerate(planes):
            for opening_key in plane["openings"].keys():
                openings.append(plane["openings"][opening_key]["center"])
                order.append([idx, opening_key])

        if len(openings) > 0:

            openings_features = np.asarray(openings).reshape((-1, 2))

            clustering = DBSCAN(eps=0.5, min_samples=1).fit(openings_features)
            labels = clustering.labels_.astype(np.float_)
            max_label = labels.max()

            for idx in range(max_label.astype(np.int_) + 1):
                keys = np.where(labels == idx)[0]
                keys = [order[key] for key in keys]
                openings_to_fuse = [planes[key[0]]["openings"][str(key[1])] for key in keys]
                fused_openings[str(idx)] = {}
                fused_openings[str(idx)]["max_bound"] = openings_to_fuse[0]["max_bound"]
                fused_openings[str(idx)]["min_bound"] = openings_to_fuse[0]["min_bound"]
                for opening_id in range(len(openings_to_fuse)):
                    fused_openings[str(idx)]["max_bound"] = np.max(
                        np.vstack((fused_openings[str(idx)]["max_bound"],
                                   openings_to_fuse[opening_id]["max_bound"])), axis=0)
                    fused_openings[str(idx)]["max_bound"] = np.min(
                        np.vstack((fused_openings[str(idx)]["max_bound"],
                                   fused_plane["max_bound"])), axis=0)
                    fused_openings[str(idx)]["min_bound"] = np.min(
                        np.vstack((fused_openings[str(idx)]["min_bound"],
                                   openings_to_fuse[opening_id]["min_bound"])), axis=0)
                    fused_openings[str(idx)]["min_bound"] = np.max(
                        np.vstack((fused_openings[str(idx)]["min_bound"],
                                   fused_plane["min_bound"])), axis=0)

                fused_openings[str(idx)]["center"] = \
                    np.asarray([0.5 * (fused_openings[str(idx)]["max_bound"][0] +
                                       fused_openings[str(idx)]["min_bound"][0]),
                                0.5 * (fused_openings[str(idx)]["max_bound"][1] +
                                       fused_openings[str(idx)]["min_bound"][1])])

        return fused_openings
            

    def pps_from_robot_to_map(self, pose_robot, pps_mean, pps_cov):
        """

        :param pose_robot: transformation robot-map with covariance matrix
        :param pps_mean: mean of the plane representation in the PPS w.r.t. the robot frame
        :param pps_cov: covariance matrix of the plane representation in the PPS w.r.t. the robot frame
        :return: mean and covariance matrix of the plane representation in the PPS w.r.t. the world frame
        """

        x = pose_robot.pose.pose.position.x
        y = pose_robot.pose.pose.position.y
        theta = euler_from_quaternion((pose_robot.pose.pose.orientation.x, pose_robot.pose.pose.orientation.y,
                                       pose_robot.pose.pose.orientation.z, pose_robot.pose.pose.orientation.w))[-1]

        pose_cov = np.asarray(pose_robot.pose.covariance).reshape((6, 6))
        pose_cov = np.asarray([[pose_cov[0, 0], pose_cov[0, 1], pose_cov[0, -1]],
                               [pose_cov[1, 0], pose_cov[1, 1], pose_cov[1, -1]],
                               [pose_cov[-1, 0], pose_cov[-1, 1], pose_cov[-1, -1]]], dtype=np.float64)


        plane_cov = np.asarray([[pps_cov[0, 0], pps_cov[0, -1]], [pps_cov[-1, 0], pps_cov[-1, -1]]], dtype=np.float64)

        pps_mean_global = np.asarray([[((pps_mean[0] + theta))],
                                      [pi / 2],
                                      [(pps_mean[-1] + np.linalg.norm(np.asarray([x, y])) * cos(
                                          (theta + pps_mean[0] - atan2(y, x))))]],
                                     dtype=np.float64)

        j_pose = self.jacobian_pose(x, y, theta, pps_mean[0])
        j_plane = self.jacobian_plane(x, y, theta, pps_mean[0])

        cov_pose_global = np.matmul(np.matmul(j_pose, pose_cov), j_pose.T)
        cov_plane_global = np.matmul(np.matmul(j_plane, plane_cov), j_plane.T)

        pps_cov_global = cov_pose_global + cov_plane_global

        # Maximum covariances to reject an observation
        # Good covariances: 0.01 and 0.1
        # Less restrictive: 0.05 and 0.15

        pps_cov_global = np.asarray(
            [[pps_cov_global[0, 0], 0., pps_cov_global[0, -1]], [0., 0.0000001, 0.],
             [pps_cov_global[-1, 0], 0., pps_cov_global[-1, -1]]],
            dtype=np.float64)

        return pps_mean_global.reshape(3, ), pps_cov_global

    @staticmethod
    def jacobian_pose(x, y, theta, alpha):
        """

        :param x: x-position of the robot
        :param y: y-position of the robot
        :param theta: theta-orientation of the robot
        :param alpha: alpha of the plane in the PPS
        :return:
        """

        norm = np.linalg.norm(np.asarray([x, y]))
        c = cos(alpha + theta - atan2(y, x))
        s = sin(alpha + theta - atan2(y, x))

        return np.asarray([[0.0, 0.0, 1.0],
                           [(x * c - y * s) / norm, (x * s + y * c) / norm, -norm * s]], dtype=np.float64)

    @staticmethod
    def jacobian_plane(x, y, theta, alpha):
        """

        :param x: x-position of the robot
        :param y: y-position of the robot
        :param theta: theta-orientation of the robot
        :param alpha: alpha of the plane in the PPS
        :return:
        """

        norm = np.linalg.norm(np.asarray([x, y]))
        s = sin(alpha + theta - atan2(y, x))

        return np.asarray([[1.0, 0.0], [-norm * s, 1.0]], dtype=np.float64)

    ####################################################################################################################
    ############################################## Statistical Distances ###############################################
    ####################################################################################################################

    @staticmethod
    def bhattacharyya_distance_features(x1, x2):
        """

        :param x1: features of the plane 1
        :param x2: features of the plane 2
        :return: Bhattacharyya distance between planes 1 and 2
        """

        mean1 = x1[0:3].astype(np.float_).reshape((3,))
        cov1 = x1[3:12].astype(np.float_).reshape((3, 3))
        mean2 = x2[0:3].astype(np.float_).reshape((3,))
        cov2 = x2[3:12].astype(np.float_).reshape((3, 3))

        mixed_cov = 0.5 * (cov1 + cov2)
        inv_mixed_cov = np.linalg.inv(mixed_cov)

        det1 = np.linalg.det(cov1)
        det2 = np.linalg.det(cov2)
        det_mixed = np.linalg.det(mixed_cov)


        ### ToDo: DESCOMENTAR CUANDO BORRE LO DE ABAJO, PQ ESTA FUNCIONA PA RADIANES
        mean_diff = np.asarray([min((2 * pi) - abs(mean1[0] - mean2[0]), abs(mean1[0] - mean2[0])),
                                min((2 * pi) - abs(mean1[1] - mean2[1]), abs(mean1[1] - mean2[1])),
                                (mean1[2] - mean2[2])]).reshape((3,))
        mean_diff = np.asarray([min((360.) - abs(mean1[0] - mean2[0]), abs(mean1[0] - mean2[0])),
                                min((360.) - abs(mean1[1] - mean2[1]), abs(mean1[1] - mean2[1])),
                                (mean1[2] - mean2[2])]).reshape((3,))
                    
        return (np.matmul(np.matmul(mean_diff.T, inv_mixed_cov), mean_diff) / 8) + \
               (0.5 * np.log(det_mixed / np.sqrt(det1 * det2)))

    @staticmethod
    def bhattacharyya_distance_features_separated(x1, x2):
        """

        :param x1: features of the plane 1
        :param x2: features of the plane 2
        :return: Bhattacharyya distance between planes 1 and 2
        """

        bhatta_1d = lambda m1, s1, m2, s2: 0.25 * ((m1 - m2)**2 / (s1 + s2)) + 0.5 * np.log((s1 + s2) / (2*np.sqrt(s1)*np.sqrt(s2)))

        mean1 = x1[0:3].astype(np.float_).reshape((3,))
        cov1 = x1[3:12].astype(np.float_).reshape((3, 3))
        mean2 = x2[0:3].astype(np.float_).reshape((3,))
        cov2 = x2[3:12].astype(np.float_).reshape((3, 3))

        d_angle = bhatta_1d(mean1[0], cov1[0,0], mean2[0], cov2[0,0])
        d_distance = bhatta_1d(mean1[2], cov1[2,2], mean2[2], cov2[2,2])

        
        return d_angle + d_distance

    @staticmethod
    def error_free_distance(x1, x2):
        """

        :param x1: features of the plane 1
        :param x2: features of the plane 2
        :return: Bhattacharyya distance between planes 1 and 2
        """

        mean1 = x1[0:3].astype(np.float_).reshape((3,))
        mean2 = x2[0:3].astype(np.float_).reshape((3,))
        center1 = np.average(x1[12:].astype(np.float).reshape((4,2)), axis=0)
        center2 = np.average(x2[12:].astype(np.float).reshape((4,2)), axis=0)

        normal2 = np.array([ np.cos(mean2[0]) * np.sin(mean2[1]), np.sin(mean2[0]) * np.sin(mean2[1]), np.cos(mean2[1])])

        d_angle = 0. if abs(min((2 * pi) - abs(mean1[0] - mean2[0]), abs(mean1[0] - mean2[0]))) < 0.2 else 100.

        d_distance = abs((center1 - center2).dot(normal2[:2]))

        d_distance = 0. if d_distance < 0.15 else 100.

        return d_angle + d_distance

    ####################################################################################################################
    ############################################## Global Map Refinement ###############################################
    ####################################################################################################################

    def refine_wall(self, wall, intersections, max_d=0.8):
        """

        :param wall: wall to refine
        :param intersections: intersection points of the wall
        :param max_d: maximum distance between each wall and their intersection to accept their intersection
        :return:
        """
        # max_d = 0 removes the intersection completation

        wall_corners = self.get_plane_corners(wall)
        wall_corners = [wall_corners[0], wall_corners[-1]]

        refine_max = []
        refine_min = []

        for intersection in intersections:
            dist, corner, _ = self.min_distance_to_intersection(wall_corners, intersection)

            if dist < max_d:
                dist_max = np.linalg.norm(wall['max_bound'][:2] - corner[:2])
                dist_min = np.linalg.norm(wall['min_bound'][:2] - corner[:2])

                if dist_max < dist_min:
                    angle1 = atan2(intersection[1] - wall_corners[0][1], intersection[0] - wall_corners[0][0])
                    angle2 = atan2(intersection[1] - wall_corners[1][1], intersection[0] - wall_corners[1][0])
                    if min((2 * pi) - abs(angle1 - angle2), abs(angle1 - angle2)) < 0.05:
                        refine_max.append([intersection[:2], False, dist_max])
                    else:
                        refine_max.append([intersection[:2], True, dist_max])

                else:
                    angle1 = atan2(intersection[1] - wall_corners[0][1], intersection[0] - wall_corners[0][0])
                    angle2 = atan2(intersection[1] - wall_corners[1][1], intersection[0] - wall_corners[1][0])
                    if min((2 * pi) - abs(angle1 - angle2), abs(angle1 - angle2)) < 0.05:
                        refine_min.append([intersection[:2], False, dist_min])
                    else:
                        refine_min.append([intersection[:2], True, dist_min])

        if refine_max:
            if False in [i[1] for i in refine_max]:
                refine_max = [[i[0], i[-1]] for i in refine_max if not i[1]]
                wall['max_bound'][:2] = sorted(refine_max, key=lambda x: x[1])[-1][0]
            else:
                wall['max_bound'][:2] = sorted(refine_max, key=lambda x: x[1])[0][0]

        if refine_min:
            if False in [i[1] for i in refine_min]:
                refine_min = [[i[0], i[-1]] for i in refine_min if not i[1]]
                wall['min_bound'][:2] = sorted(refine_min, key=lambda x: x[1])[-1][0]
            else:
                wall['min_bound'][:2] = sorted(refine_min, key=lambda x: x[1])[0][0]

        return wall

    def global_refinement(self, map, max_d=0.8):
        """

        :param map: dictionary representing the current 3D floorplan
        :param max_d: maximum distance between each wall and their intersection to accept their intersection
        :return: dictionary representing the refined 3D floorplan
        """

        max_wall_height = 0
        intersections = {}

        map_cp = deepcopy(map)

        for wall_key in map_cp.keys():

            if map[wall_key]['n_detections'] < 2:
                map.pop(wall_key)

            else:
                max_wall_height = max(max_wall_height, map[wall_key]['max_bound'][-1])

        map_cp = deepcopy(map)
        for c in set(itertools.combinations(map_cp.keys(), 2)):

            intersection = self.plane_intersect(map[c[0]], map[c[1]])

            if intersection is not None:
                wall_corners = self.get_plane_corners(map[c[0]])
                dist1, _, inside1 = self.min_distance_to_intersection([wall_corners[0], wall_corners[-1]], intersection)
                wall_corners = self.get_plane_corners(map[c[1]])
                dist2, _, inside2 = self.min_distance_to_intersection([wall_corners[0], wall_corners[-1]], intersection)

                if (not inside1 and dist1 > max_d) or (not inside2 and dist2 > max_d):
                    continue

                if c[0] in intersections.keys():
                    intersections[c[0]].append(intersection)
                else:
                    intersections[c[0]] = [intersection]

                if c[1] in intersections.keys():
                    intersections[c[1]].append(intersection)
                else:
                    intersections[c[1]] = [intersection]

        for k in intersections.keys():
            map[k] = self.refine_wall(map[k], intersections[k])

        for wall_key in map.keys():

            map[wall_key]['max_bound'][-1] = max_wall_height
            map[wall_key]['min_bound'][-1] = 0.

            for o_key in map[wall_key]['openings'].keys():

                if map[wall_key]['openings'][o_key]['min_bound'][-1] < 0.5:
                    map[wall_key]['openings'][o_key]['min_bound'][-1] = 0.

        return map

    ####################################################################################################################
    ################################################## Visualization ###################################################
    ####################################################################################################################

    def create_msg_walls_markers(self, walls, mapframe_name="map"):
        """

        :param walls: dictionary including the current 3D floorplan
        :return: set of Markers msg to visualize the 3D floorplan in RVIZ
        """
        msg = MarkerArray()
        header = Header()
        header.frame_id = mapframe_name
        header.stamp = rospy.Time.now()
        remove_marker = Marker()
        remove_marker.header = Header()
        remove_marker.action = remove_marker.DELETEALL
        msg.markers.append(remove_marker)
        for idx, key in enumerate(walls.keys()):
            plane = Marker()
            plane.id = idx
            plane.action = plane.ADD
            plane.type = plane.TRIANGLE_LIST
            plane.header = header
            plane.scale.x = 1.0
            plane.scale.y = 1.0
            plane.scale.z = 1.0
            plane_color = walls[key]["color"]
            plane.pose.orientation.w = 1.0
            wall_corners = self.get_plane_corners_with_full_openings_triangulation(walls[key])
            for i in range(len(wall_corners)):
                plane.points.append(Point(*wall_corners[i]))
                plane.colors.append(plane_color)

            msg.markers.append(plane)

        return msg

    @staticmethod
    def point_cloud_visualization_rviz(pcd_o3d, header_img, mapframe_name="map"):
        """

        :param pcd_o3d: point cloud in open3D
        :param header_img: header of the image from which the point cloud is computed
        :return: msg to publish to visualize the point cloud in RVIZ
        """

        header = Header()
        header.stamp = header_img.stamp
        header.frame_id = mapframe_name
        points = np.asarray(pcd_o3d.points)

        if not pcd_o3d.colors:  # XYZ only
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            cloud_data = points
        else:  # XYZ + RGB
            fields = [
                         PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                         PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                         PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                     ] + \
                     [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]
            # -- Change rgb color from "three float" to "one 24-byte int"
            # 0x00FFFFFF is white, 0x00000000 is black.
            colors = np.floor(np.asarray(pcd_o3d.colors) * 255)  # nx3 matrix
            colors = colors[:, 0] * 2 ** 16 + colors[:, 1] * 2 ** 8 + colors[:, 2]
            cloud_data = np.c_[points, colors]

        msg = pc2.create_cloud(header, fields, cloud_data)

        return msg

    ####################################################################################################################
    ############################################# Basic Geometry Functions #############################################
    ####################################################################################################################

    def isect_line_plane_v3_4d(self, p0, p1, plane, epsilon=1e-6):
        """

        :param p0: first point which defines a line
        :param p1: second point which defines a line
        :param plane: plane to check if its intersected by the line
        :param epsilon: error threshold - do not change
        :return: intersection (if exists)
        """
        u = self.sub_v3v3(p1, p0)
        dot = self.dot_v3v3(plane, u)

        if abs(dot) > epsilon:
            # Calculate a point on the plane
            # (divide can be omitted for unit hessian-normal form).
            p_co = self.mul_v3_fl(plane, -plane[3] / self.len_squared_v3(plane))

            w = self.sub_v3v3(p0, p_co)
            fac = -self.dot_v3v3(plane, w) / dot
            u = self.mul_v3_fl(u, fac)
            return self.add_v3v3(p0, u)

        return None

    # Generic functions
    @staticmethod
    def add_v3v3(v0, v1):
        """

        :param v0: vector 0
        :param v1: vector 1
        :return: sum of vector 0 and 1
        """
        return (
            v0[0] + v1[0],
            v0[1] + v1[1],
            v0[2] + v1[2],
        )

    @staticmethod
    def sub_v3v3(v0, v1):
        """

        :param v0: vector 0
        :param v1: vector 1
        :return: subtraction of vector 0 and 1
        """
        return (
            v0[0] - v1[0],
            v0[1] - v1[1],
            v0[2] - v1[2],
        )

    @staticmethod
    def dot_v3v3(v0, v1):
        """

        :param v0: vector 0
        :param v1: vector 1
        :return: dot of vector 0 and 1
        """
        return (
                (v0[0] * v1[0]) +
                (v0[1] * v1[1]) +
                (v0[2] * v1[2])
        )

    @staticmethod
    def mul_v3_fl(v0, f):
        """

        :param v0: vector 0
        :param f: factor
        :return: scale of vector 0 by factor f
        """
        return (
            v0[0] * f,
            v0[1] * f,
            v0[2] * f,
        )

    def len_squared_v3(self, v0):
        """

        :param v0: vector 0
        :return: length squared of vector 0
        """
        return self.dot_v3v3(v0, v0)

