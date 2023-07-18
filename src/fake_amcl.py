#! /usr/bin/env python3

import rospy
import tf2_ros

from geometry_msgs.msg import PoseWithCovarianceStamped, Point

class FakeAMCL(object):

    def __init__(self):

        # Control
        self._publishRate = 250
        
        # Mode (tf: receives the gt through tfs / topic: receives the gt through a topic)
        self.mode = self.load_param('~mode', "topic") # topic or tf
        self.topic_name = self.load_param('~topic_name', "gt")
        self.baselink_name = self.load_param('~baselink_name', "base_link")
        self.world_name = self.load_param('~world_name', "map")

        # Publishers
        self._pose_pub = rospy.Publisher('/amcl_pose', PoseWithCovarianceStamped, queue_size=100)

        if self.mode == "topic":
            rospy.Subscriber(self.topic_name, tf2_ros.TFMessage, self.gt_callback)
        else:
            self._tfBuffer = tf2_ros.Buffer()
            tf2_ros.TransformListener(self._tfBuffer)

        self._started = False
        self._first_pose = None

    def create_pose_msg(self, data):

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = data.header.stamp
        msg.header.frame_id = self.world_name

        msg.pose.pose.position = Point()
        msg.pose.pose.position.x = data.transform.translation.x
        msg.pose.pose.position.y = data.transform.translation.y
        msg.pose.pose.position.z = data.transform.translation.z
        msg.pose.pose.orientation = data.transform.rotation
        msg.pose.covariance = [0.0000,0.0,0.0,0.0,0.0,0.0,
                               0.0,0.0000,0.0,0.0,0.0,0.0,
                               0.0,0.0,0.0,0.0,0.0,0.0,
                               0.0,0.0,0.0,0.0,0.0,0.0,
                               0.0,0.0,0.0,0.0,0.0,0.0,
                               0.0,0.0,0.0,0.0,0.0,0.00000]
        
        self._pose_pub.publish(msg)


    # Node operation
    def run(self):

        rate = rospy.Rate(self._publishRate)
        new_pose = None

        while not rospy.is_shutdown():
            try:
                if self.mode == "tf":
                    new_pose = self._tfBuffer.lookup_transform(self.world_name,
                                                                self.baselink_name,
                                                                rospy.Time())

            except:
                pass
            if new_pose != None:
                self.create_pose_msg(new_pose)


            rate.sleep()

    # Callbacks definition

    def gt_callback(self, data):
        
        self.create_pose_msg(data.transforms[0])

    ####################################################################################################################
    ################################################### Static Methods #################################################
    ####################################################################################################################

    @staticmethod
    def load_param(param, default=None):
        new_param = rospy.get_param(param, default)
        rospy.loginfo("[Sigma-FP] %s: %s", param, new_param)
        return new_param

def main():
    rospy.init_node('FakeAMCL', anonymous=True)
    node = FakeAMCL()
    node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass