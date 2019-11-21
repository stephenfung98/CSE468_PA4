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
import os
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

referenceImg = cv2.imread('/home/apollo/catkin_ws/src/CSE468_PA4/target2.png',0)
referenceImg1 = cv2.imread('/home/apollo/catkin_ws/src/CSE468_PA4/target3.png',0)
referenceImg2 = cv2.imread('/home/apollo/catkin_ws/src/CSE468_PA4/target4.png',0)

# cv2.imshow('img0', referenceImg)
# cv2.waitKey(0)
isFound = False
count = 0


def callback(data):
    global referenceImg, count, isFound
    if not isFound:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        img_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)


        img1 = img_gray          # queryImage
        img2 = referenceImg # trainImage
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.9*n.distance:
                good.append([m])
        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        print(len(good))

        if len(good) >= 60:
            isFound = True
            # os.system("rosnode kill /turtlebot_teleop_keyboard")
            plt.imshow(img3),plt.show()

        
def listener():
    rospy.init_node('yInt', anonymous=True)

    rospy.Subscriber("/camera/rgb/image_raw", Image, callback,  queue_size = 1)

    rospy.spin()

if __name__ == '__main__':
    listener()