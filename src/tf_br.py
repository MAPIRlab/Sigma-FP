#! /usr/bin/env python3

import rospy
import time
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Quaternion

class BroadcastingTFs(object):

    def __init__(self):

        # Control
        self._publishRate = 0.5
        self._start = False

        # Suscribers
        rospy.Subscriber("tf_static", TFMessage, self._new_tf)

        # Publishers
        self._pub = rospy.Publisher('/tf', TFMessage, queue_size=10)

        # Constants
        self._tf_statics = None

        # Variables

    def _new_tf(self, data):
        self._tf_statics = data
        self._start = True

    # Node operation
    def run(self):
        rate = rospy.Rate(self._publishRate)
        while not rospy.is_shutdown():
            if self._start:
                t = rospy.Time.now()
                if hasattr(self._tf_statics, 'header') and hasattr(self._tf_statics.header, 'stamp'):
                    self._tf_statics.header.stamp = t
            

                self._pub.publish(self._tf_statics)
            rate.sleep()

    # Callbacks functions definition


def main():
    rospy.init_node('BroadcastingTFs', anonymous=True)
    node = BroadcastingTFs()
    node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass