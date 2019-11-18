#!/usr/bin/env python
from __future__ import print_function
import rospy, time
import math
from math import atan, degrees, cos, sin
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
import sys, time
import numpy as np
# from scipy.ndimage import filters
import cv2
import roslib
import rospy
from sensor_msgs.msg import CompressedImage
import roslib
# roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from matplotlib import pyplot as plt

bridge = CvBridge()

referenceImg = cv2.imread('/home/apollo/catkin_ws/src/CSE468_PA4/target.png',0)
# cv2.imshow('img0', referenceImg)
# cv2.waitKey(0)
isFound = False
count = 0


def callback(data):
    global referenceImg, count
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

    surf = cv2.xfeatures2d.SURF_create(400)
    kp1,des1 = surf.detectAndCompute(cv_image,None)
    kp2,des2 = surf.detectAndCompute(referenceImg,None)

    bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
    matches = bf.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    print(len(good))

        
def listener():
    rospy.init_node('yInt', anonymous=True)

    rospy.Subscriber("/camera/rgb/image_raw", Image, callback,  queue_size = 1)

    rospy.spin()

if __name__ == '__main__':
    listener()