import numpy as np
from math import sin, cos

import tf.transformations as tr

from geometry_msgs.msg import PoseStamped, Transform, TransformStamped, Quaternion
from geometry_msgs.msg import Pose


class Transformations(object):

    def pose_to_pq(self, msg):
        """Convert a C{geometry_msgs/Pose} into position/quaternion np arrays

        :param msg: ROS message to be converted
        :return:
          - p: position as a np.array
          - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        p = np.array([msg.position.x, msg.position.y, msg.position.z])
        q = np.array([msg.orientation.x, msg.orientation.y,
                      msg.orientation.z, msg.orientation.w])
        return p, q

    def pose_stamped_to_pq(self, msg):
        """Convert a C{geometry_msgs/PoseStamped} into position/quaternion np arrays

        :param msg: ROS message to be converted
        :return:
          - p: position as a np.array
          - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        return self.pose_to_pq(msg.pose)

    def transform_to_pq(self, msg):
        """Convert a C{geometry_msgs/Transform} into position/quaternion np arrays

        :param msg: ROS message to be converted
        :return:
          - p: position as a np.array
          - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
        q = np.array([msg.rotation.x, msg.rotation.y,
                      msg.rotation.z, msg.rotation.w])
        return p, q

    def transform_stamped_to_pq(self, msg):
        """Convert a C{geometry_msgs/TransformStamped} into position/quaternion np arrays

        :param msg: ROS message to be converted
        :return:
          - p: position as a np.array
          - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        return self.transform_to_pq(msg.transform)

    def msg_to_se3(self, msg):
        """Conversion from geometric ROS messages into SE(3)

        :param msg: Message to transform. Acceptable types - C{geometry_msgs/Pose}, C{geometry_msgs/PoseStamped},
        C{geometry_msgs/Transform}, or C{geometry_msgs/TransformStamped}
        :return: a 4x4 SE(3) matrix as a numpy array
        @note: Throws TypeError if we receive an incorrect type.
        """
        if isinstance(msg, Pose):
            p, q = self.pose_to_pq(msg)
        elif isinstance(msg, PoseStamped):
            p, q = self.pose_stamped_to_pq(msg)
        elif isinstance(msg, Transform):
            p, q = self.transform_to_pq(msg)
        elif isinstance(msg, TransformStamped):
            p, q = self.transform_stamped_to_pq(msg)
        else:
            raise TypeError("Invalid type for conversion to SE(3)")
        norm = np.linalg.norm(q)
        if np.abs(norm - 1.0) > 1e-3:
            raise ValueError(
                "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                    str(q), np.linalg.norm(q)))
        elif np.abs(norm - 1.0) > 1e-6:
            q = q / norm
        g = tr.quaternion_matrix(q)
        g[0:3, -1] = p
        return g

    @staticmethod
    def quaternion_multiply(quaternion0, quaternion1):
        """

        :param quaternion0: quaternion 0
        :param quaternion1: quaternion 1
        :return: multiplication of quaternion 0 and 1
        """
        w0 = quaternion0.w
        x0 = quaternion0.x
        y0 = quaternion0.y
        z0 = quaternion0.z

        w1 = quaternion1.w
        x1 = quaternion1.x
        y1 = quaternion1.y
        z1 = quaternion1.z

        return Quaternion(w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1, w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                          w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1, w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1)

    @staticmethod
    def x_rotation(theta):
        """

        :param theta: rotation angle in X-axis
        :return: matrix of transformation representing the rotation of theta in X-axis
        """
        return np.asarray([[1, 0, 0], [0, cos(theta), -sin(theta)], [0, sin(theta), cos(theta)]])

    @staticmethod
    def y_rotation(theta):
        """

        :param theta: rotation angle in Y-axis
        :return: matrix of transformation representing the rotation of theta in Y-axis
        """
        return np.asarray([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])

    @staticmethod
    def z_rotation(theta):
        """

        :param theta: rotation angle in Z-axis
        :return: matrix of transformation representing the rotation of theta in Z-axis
        """
        return np.asarray([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
