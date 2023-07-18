import os
import numpy as np
import cv2

from modules.external_transformations import *

class DataSaver(object):

    # Initialization
    def __init__(self, map_path, save_path, width, height, fx, fy, cx, cy):
        self.map_path = map_path
        self.save_path = save_path

        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        origin, xaxis, yaxis, zaxis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        self.Ry = rotation_matrix(np.radians(90), yaxis)
        self.Rz = rotation_matrix(np.radians(-90), zaxis)

        self._k_localization = 0
        self._k_mapping = 0

        self.loc_dir = self.save_path + "colmap/localization/"
        self.map_dir = self.save_path + "colmap/mapping/"

        self.normals_r = None
        self.center_r = None
        self.maxb_r = None
        self.min_r = None

        self.use_current_map = False
        

    def create_directory(self):

        if os.path.exists(self.save_path):

            if not os.path.exists(self.map_dir):
                os.makedirs(self.map_dir)

            if not os.path.exists(self.loc_dir):
                os.makedirs(self.loc_dir)

            if not os.path.exists(self.map_dir + "input_model/"):
                os.mkdir(self.map_dir + "input_model/")

            if not os.path.exists(self.map_dir + "input_model/"):
                os.mkdir(self.map_dir + "input_model/")

            if not os.path.exists(self.map_dir + "images/"):
                os.mkdir(self.map_dir + "images/")

            if not os.path.exists(self.map_dir + "masks/"):
                os.mkdir(self.map_dir + "masks/")

            if not os.path.exists(self.map_dir + "masks_binary/"):
                os.mkdir(self.map_dir + "masks_binary/")

            if not os.path.exists(self.map_dir + "masks_ceilingfloor/"):
                os.mkdir(self.map_dir + "masks_ceilingfloor/")

            if not os.path.exists(self.loc_dir + "images/"):
                os.mkdir(self.loc_dir + "images/")

            if not os.path.exists(self.loc_dir + "masks/"):
                os.mkdir(self.loc_dir + "masks/")

            if not os.path.exists(self.loc_dir + "masks_binary/"):
                os.mkdir(self.loc_dir + "masks_binary/")

            if not os.path.exists(self.loc_dir + "masks_ceilingfloor/"):
                os.mkdir(self.loc_dir + "masks_ceilingfloor/")
            
            if not os.path.exists(self.loc_dir + "depths/"):
                os.mkdir(self.loc_dir + "depths/")

            if not os.path.exists(self.map_dir + "input_model/points3D.txt"):
                with open(self.map_dir + "input_model/points3D.txt", "w"):
                    pass

            if not os.path.exists(self.map_dir + "input_model/cameras.txt"):
                with open(self.map_dir + "input_model/cameras.txt", "w") as f:
                    pass

            if not os.path.exists(self.map_dir + "input_model/images.txt"):
                with open(self.map_dir + "input_model/images.txt", "w") as f:
                    pass

            if not os.path.exists(self.loc_dir + "images.txt"):
                with open(self.loc_dir + "images.txt", "w") as f:
                    pass

            if not os.path.exists(self.loc_dir + "cameras.txt"):
                with open(self.loc_dir + "images.txt", "w") as f:
                    pass

            return True
        else:

            return False

    def load_global_map(self):
        
        try:

            reference_map = np.load(self.map_path, allow_pickle=True, encoding="latin1").item()

            self.normals_r = np.zeros((len(reference_map.keys()),3), dtype=float)
            self.centers_r = np.zeros((len(reference_map.keys()),3), dtype=float)
            self.maxb_r = np.zeros((len(reference_map.keys()),3), dtype=float)
            self.minb_r = np.zeros((len(reference_map.keys()),3), dtype=float)
            self.idx_init = 0

            for idx, k in enumerate(reference_map.keys()):

                plane = reference_map[k]["mean"]

                self.normals_r[idx,:] = [ np.cos(plane[0]) * np.sin(plane[1]), np.sin(plane[0]) * np.sin(plane[1]), np.cos(plane[1])]
                p1 = (reference_map[k]["max_bound"] + reference_map[k]["min_bound"]) / 2.0
                self.centers_r[idx,:] = p1 + (plane[2] - sum(p1 * self.normals_r[idx,:])) / sum(self.normals_r[idx,:] * self.normals_r[idx,:]) * self.normals_r[idx,:]
                self.maxb_r[idx,:] = reference_map[k]["max_bound"].reshape((1,3)) + 0.15
                self.minb_r[idx,:] = reference_map[k]["min_bound"].reshape((1,3)) - 0.15

            success = self.create_directory()
            if success:
                return True
            else:
                return False

        except:

            return False

    def add_current_map(self, map):

        self.use_current_map = True

        self.normals_r_cp = self.normals_r.copy()
        self.centers_r_cp = self.centers_r.copy()
        self.maxb_r_cp = self.maxb_r.copy()
        self.minb_r_cp = self.minb_r.copy()

        for idx, k in enumerate(map.keys()):

            plane = map[k]["mean"]

            n = np.array([ np.cos(plane[0]) * np.sin(plane[1]), np.sin(plane[0]) * np.sin(plane[1]), np.cos(plane[1])])
            c = (map[k]["max_bound"] + map[k]["min_bound"]) / 2.0
            c += (plane[2] - sum(c * n)) / sum(n * n) * n
            self.normals_r_cp = np.vstack((self.normals_r_cp, n)) 
            self.centers_r_cp = np.vstack((self.centers_r_cp, c.reshape((1,3))))
            self.maxb_r_cp = np.vstack((self.maxb_r_cp, map[k]["max_bound"].reshape((1,3)) + 0.15))
            self.minb_r_cp = np.vstack((self.minb_r_cp, map[k]["min_bound"].reshape((1,3)) - 0.15))

    def compute_planes_mask(self, xyz_img_global, residuals = False, limits = True):
        
        planes_mask = np.full((xyz_img_global.shape[0], xyz_img_global.shape[1]), False)

        if not self.use_current_map:

            self.normals_r_cp = self.normals_r.copy()
            self.centers_r_cp = self.centers_r.copy()
            self.maxb_r_cp = self.maxb_r.copy()
            self.minb_r_cp = self.minb_r.copy()

        for i in range(self.normals_r_cp.shape[0]):

            if residuals:
                d_r = -np.sum(self.normals_r_cp[i,:] * self.centers_r_cp)
                res_img = np.abs(np.sum(self.normals_r_cp[i,:] * xyz_img_global, axis = 2) + d_r)
                res_img = res_img < 0.05
            if limits:
                in_limits = np.logical_and(xyz_img_global[:,:,:2] > self.minb_r_cp[i,:2], xyz_img_global[:,:,:2] < self.maxb_r_cp[i,:2])
                in_limits = np.logical_and(in_limits[:,:,0], in_limits[:,:,1])

            if residuals and limits:
                planes_mask[np.logical_and(res_img, in_limits)] = True
            elif residuals and not limits:
                planes_mask[res_img] = True
            elif not residuals and limits:
                planes_mask[in_limits] = True
            else:
                print("[DATASAVER] None method selected for plane mask computation.")

        return planes_mask
    
    def compute_instance_planes_mask(self, xyz_img_global, residuals = False, limits = True, ceilingandfloor=True):

        # ToDo: remove, just a try!
        #zx = cv2.Sobel(xyz_img_global[:,:,2], cv2.CV_64F, 1, 0, ksize=7)     
        #zy = cv2.Sobel(xyz_img_global[:,:,2], cv2.CV_64F, 0, 1, ksize=7)
    
        #normals = np.dstack((-zx, -zy, np.ones_like(xyz_img_global[:,:,2])))
        #normals[:, :, 2] /= np.linalg.norm(normals, axis=2)
        
        #normals_mask = (normals[:,:,2] < 0.3)

        ###
        
        planes_mask = np.zeros((xyz_img_global.shape[0], xyz_img_global.shape[1]), dtype=np.uint8)

        if not self.use_current_map:

            self.normals_r_cp = self.normals_r.copy()
            self.centers_r_cp = self.centers_r.copy()
            self.maxb_r_cp = self.maxb_r.copy()
            self.minb_r_cp = self.minb_r.copy()

        global_residuals = np.ones((xyz_img_global.shape[0], xyz_img_global.shape[1]), dtype=float)

        for i in range(self.normals_r_cp.shape[0]):

            if residuals:
                d_r = -np.sum(self.normals_r_cp[i,:] * self.centers_r_cp[i,:])
                res_img = np.abs(np.sum(self.normals_r_cp[i,:] * xyz_img_global, axis = 2) + d_r)
                global_residuals[res_img < global_residuals] = res_img[res_img < global_residuals].copy()
                res_img = np.logical_and(res_img < 0.15, res_img <= global_residuals)
                
            if limits:
                in_limits = np.logical_and(xyz_img_global[:,:,:2] > self.minb_r_cp[i,:2], xyz_img_global[:,:,:2] < self.maxb_r_cp[i,:2])
                in_limits = np.logical_and(in_limits[:,:,0], in_limits[:,:,1])

            if residuals and limits:
                planes_mask[np.logical_and(res_img, in_limits)] = i + 1
            elif residuals and not limits:
                planes_mask[res_img] = i + 1
                #planes_mask[np.logical_and(res_img, normals_mask)] = i + 1
            elif not residuals and limits:
                planes_mask[in_limits] = i + 1
            else:
                print("[DATASAVER] None method selected for plane mask computation.")
        
        if ceilingandfloor:
            planes_mask_ceilingfloor = planes_mask.copy()
            #xy_norm_ok = np.linalg.norm(xyz_img_global[:,:,:2], axis=2) < 7.0

            planes_mask_ceilingfloor[xyz_img_global[:,:,2] < 0.05] = self.normals_r_cp.shape[0] + 1
            planes_mask_ceilingfloor[np.logical_and(xyz_img_global[:,:,2] > 2.85, xyz_img_global[:,:,2] < 3.20)] = self.normals_r_cp.shape[0] + 2

            return planes_mask, planes_mask_ceilingfloor
        
        return planes_mask

    def write_data(self, img_rgb, planes_mask, pose, mode = "mapping", cam_id = 1, depth_img = None, planes_mask_ceilingfloor = None):
        
        pose_c = pose @ self.Ry @ self.Rz

        r = np.eye(4, dtype=float)
        r[:3,:3] = pose_c[:3, :3].T
        p = -r[:3,:3] @ pose_c[:3,3]
        trans = [str(x) for x in p.reshape((-1)).tolist()]
        trans = ' '.join(trans)
        q_pose = np.roll(quaternion_from_matrix(r), 1)
        q_pose = [str(x) for x in q_pose.reshape((-1)).tolist()]
        q_pose = ' '.join(q_pose)

        if mode == "mapping":
            img_path = self.map_dir + "images/image_" + "{0:05d}".format(self._k_mapping) + ".jpeg"
            mask_path = self.map_dir + "masks/image_" + "{0:05d}".format(self._k_mapping) + ".jpeg.png"

            if planes_mask_ceilingfloor is not None:

                mask_ceilingfloor_path = self.map_dir + "masks_ceilingfloor/image_" + "{0:05d}".format(self._k_mapping) + ".jpeg.png"
                cv2.imwrite(mask_ceilingfloor_path, planes_mask_ceilingfloor)
            
            # Borrar
            masks_binary_path = self.map_dir + "masks_binary/image_" + "{0:05d}".format(self._k_mapping) + ".jpeg.png"
            cv2.imwrite(masks_binary_path, (planes_mask>0).astype(np.uint8)*255)

            pose_path = self.map_dir + "input_model/images.txt"
            cv2.imwrite(img_path, cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
            cv2.imwrite(mask_path, planes_mask)
            pose_data = str(self._k_mapping + 1) + " " + q_pose + " " + trans + " " + str(cam_id) + " image_" + "{0:05d}".format(self._k_mapping) + ".jpeg"
            with open(pose_path, 'a') as f:
                    f.write(f'{pose_data}')
                    f.write("\n\n")

            if cam_id != 1 or (self._k_mapping == 0 and cam_id == 1):
                with open(self.map_dir + "input_model/cameras.txt", "a") as f:
                    if self.fx == self.fy:
                        line = str(cam_id) + " SIMPLE_PINHOLE " + str(self.width) + " " + str(self.height) + " " + str(self.fx) + " " + str(self.cx) + " " + str(self.cy)
                    else:
                        line = str(cam_id) + " PINHOLE " + str(self.width) + " " + str(self.height) + " " + str(self.fx) + " " + str(self.fy) + " " + str(self.cx) + " " + str(self.cy)
                    f.write(line)
                    f.write("\n")
            
            self._k_mapping += 1
            
        else:
            if ((planes_mask > 0).sum() / planes_mask.size) > 0.35:

                img_path = self.loc_dir + "images/query_" + "{0:05d}".format(self._k_localization) + ".jpeg"
                mask_path = self.loc_dir + "masks/query_" + "{0:05d}".format(self._k_localization) + ".jpeg.png"

                if planes_mask_ceilingfloor is not None:

                    mask_ceilingfloor_path = self.loc_dir + "masks_ceilingfloor/query_" + "{0:05d}".format(self._k_localization) + ".jpeg.png"
                    cv2.imwrite(mask_ceilingfloor_path, planes_mask_ceilingfloor)

                # Borrar
                masks_binary_path = self.loc_dir + "masks_binary/image_" + "{0:05d}".format(self._k_localization) + ".jpeg.png"
                cv2.imwrite(masks_binary_path, (planes_mask>0).astype(np.uint8)*255)

                pose_path = self.loc_dir + "images.txt"
                cv2.imwrite(img_path, cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
                cv2.imwrite(mask_path, planes_mask)
                if type(depth_img) != None:
                    depth_path = self.loc_dir + "depths/query_" + "{0:05d}".format(self._k_localization) + ".png"
                    cv2.imwrite(depth_path, (65535.*np.divide(depth_img, 10.)).astype(np.uint16))
                pose_data = str(self._k_localization + 1) + " " + q_pose + " " + trans + " " + str(cam_id) + " query_" + "{0:05d}".format(self._k_localization) + ".jpeg"
                with open(pose_path, 'a') as f:
                        f.write(f'{pose_data}')
                        f.write("\n\n")

                if cam_id != 1 or (self._k_localization == 0 and cam_id == 1):
                    with open(self.loc_dir + "cameras.txt", "a") as f:
                        if self.fx == self.fy:
                            line = str(cam_id) + " SIMPLE_PINHOLE " + str(self.width) + " " + str(self.height) + " " + str(self.fx) + " " + str(self.cx) + " " + str(self.cy)
                        else:
                            line = str(cam_id) + " PINHOLE " + str(self.width) + " " + str(self.height) + " " + str(self.fx) + " " + str(self.fy) + " " + str(self.cx) + " " + str(self.cy)
                        f.write(line)
                        f.write("\n")

                self._k_localization += 1