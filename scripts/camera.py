#!/usr/bin/python
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
import roslib
import rospy
from sensor_msgs.msg import CompressedImage
import roslib
# roslib.load_manifest('my_package')
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from matplotlib import pyplot as plt
# sys.path.insert(0, '/home/apollo/.local/lib/python2.7/site-packages')
import cv2
import os

bridge = CvBridge()

referenceImg = cv2.imread('/home/apollo/catkin_ws/src/CSE468_PA4/b.jpg',0)
referenceImg1 = cv2.imread('/home/apollo/catkin_ws/src/CSE468_PA4/target3.png',0)
referenceImg2 = cv2.imread('/home/apollo/catkin_ws/src/CSE468_PA4/target4.png',0)
model = '/home/apollo/catkin_ws/src/CSE468_PA4/weights/yolov3.weights'
config = '/home/apollo/catkin_ws/src/CSE468_PA4/cfg/yolov3.cfg'

# cv2.imshow('img0', referenceImg)
# cv2.waitKey(0)
isFound = False
count = 0
net = 0
classes = 0
output_layers = 0

def callback(data):
    global referenceImg, count, isFound, net, output_layers, classes
    if not isFound:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        # img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(cv_image, None, fx=0.4, fy=0.4)
        height, width, d = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 1.0/255.0, (416, 416), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # blobb = blob.reshape(blob.shape[2],blob.shape[3],blob.shape[1])
        # cv2.imshow('Blob',blobb)
        # cv2.waitKey(5000)

        # sort the probabilities (in descending) order, grab the index of the
        # top predicted label, and draw it on the input image
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

        for i in range(len(boxes)):
            if i in indexes:
                # color = colors[class_ids[i]]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                print(i)
                print(str(classes[class_ids[i]]))
                print(confidences[i])
    print("loop")

        
def listener():
    global config, model, net, output_layers, classes
    net = cv2.dnn.readNetFromDarknet(config, model)
    rows = open("/home/apollo/catkin_ws/src/CSE468_PA4/coco.names").read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    rospy.init_node('yInt', anonymous=True)

    rospy.Subscriber("/camera/rgb/image_raw", Image, callback,  queue_size = 1)

    rospy.spin()

if __name__ == '__main__':
    listener()