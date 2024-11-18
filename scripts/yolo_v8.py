#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import cv2
import torch
import rospy
import os
import sys
sys.path.insert(0,'/home/jy/geop3_ws/devel/lib/python3/dist-packages')

import tf
import numpy as np
from ultralytics import YOLO
from time import time
import math
from std_msgs.msg import String
import matplotlib.pyplot as plt
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from yolov8_ros_msgs.msg import BoundingBox, BoundingBoxes
from scipy.spatial.transform import Rotation as R
import scipy.spatial.transform as transform
from geometry_msgs.msg import PoseStamped

def cross_point(line1, line2):  # 计算交点函数
    x1 = line1[0]  # 取直线1的第一个点坐标
    y1 = line1[1]
    x2 = line1[2]  # 取直线1的第二个点坐标
    y2 = line1[3]
 
    x3 = line2[0]  # 取直线2的第一个点坐标
    y3 = line2[1]
    x4 = line2[2]  # 取直线2的第二个点坐标
    y4 = line2[3]
 
    if x2 - x1 == 0:  # L1 直线斜率不存在
        k1 = None
        b1 = 0
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
 
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
 
    if k1 is None and k2 is None:  # L1与L2直线斜率都不存在，两条直线均与y轴平行
        if x1 == x3:  # 两条直线实际为同一直线
            return [x1, y1]  # 均为交点，返回任意一个点
        else:
            return None  # 平行线无交点
    elif k1 is not None and k2 is None:  # 若L2与y轴平行，L1为一般直线，交点横坐标为L2的x坐标
        x = x3
        y = k1 * x * 1.0 + b1 * 1.0
    elif k1 is None and k2 is not None:  # 若L1与y轴平行，L2为一般直线，交点横坐标为L1的x坐标
        x = x1
        y = k2 * x * 1.0 + b2 * 1.0
    else:  # 两条一般直线
        if k1 == k2:  # 两直线斜率相同
            if b1 == b2:  # 截距相同，说明两直线为同一直线，返回任一点
                return [x1, y1]
            else:  # 截距不同，两直线平行，无交点
                return None
        else:  # 两直线不平行，必然存在交点
            x = (b2 - b1) * 1.0 / (k1 - k2)
            y = k1 * x * 1.0 + b1 * 1.0
    return (x, y)


def calculate_angle_between_vectors(A, B, C, D):  

    # 计算向量AB和向量CD  
    vec_AB = (B[0] - A[0], B[1] - A[1])  
    vec_CD = (D[0] - C[0], D[1] - C[1])  

    # 计算点积  
    dot_product = vec_AB[0] * vec_CD[0] + vec_AB[1] * vec_CD[1]  

    # 模长  

    magnitude_AB = math.sqrt(vec_AB[0]**2 + vec_AB[1]**2)  
    magnitude_CD = math.sqrt(vec_CD[0]**2 + vec_CD[1]**2)  

    
    # 计算夹角的余弦值  
    cosine_angle = dot_product / (magnitude_AB * magnitude_CD) 

    cosine_angle = max(-1, min(1, cosine_angle)) #余弦值超出范围会报错 

    angle_radians = math.acos(cosine_angle)   

    angle_degrees = math.degrees(angle_radians)  

      

    return angle_degrees 



class Yolo_Dect:
    def __init__(self):

        # load parameters
        weight_path = rospy.get_param('~weight_path', '')
        num_weight_path = rospy.get_param('~num_weight_path', '')
        image_topic = rospy.get_param(
            '~image_topic', '/camera/color/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov8/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = rospy.get_param('~conf', '0.5')
        vins_odomtopic = "/vins_estimator/odometry"
        door_pub_topic = "/doornum"
        self.visualize = rospy.get_param('~visualize', 'True')
        self.okk=True
        self.saveimg = False
        self.img_i = 4000

        self.vinsinit = False

        self.object =  ["309", "310", "311", "312", "313", "314", "315", "316", "317", "318", "319", "321", "322", "324", ]  
        self.coordinates = [(45.9555, -1.255), (45.9555, 1.255), (37.646, -1.255), (37.646, -1.255), (32.1735, -1.255), (32.1735, 1.255), 
                            (29.791, -1.9), (29.791, 1.9), (16.1585,-1.9), (16.1585,1.9), 
                            (13.776, -1.255), (0, -1.255), (8.3035, 1.255), (0, 1.255), ]
        
        
        self.current_attempt = None  
        self.match_count = 0 
        self.doornum_flag = False
        self.ouflag = False
        # which device will be used
        if (False):
            self.device = 'cpu'
            print("CPU!")
        else:
            self.device = 'cuda'
            print("GPU!")

        self.model = YOLO(weight_path)
        self.model.fuse()

        self.model.conf = conf
        self.color_image = Image()
        self.getImageStatus = False


        self.model_num = YOLO(num_weight_path)
        
        # Load class color
        self.classes_colors = {}

        # image subscribe
        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback,
                                         queue_size=1, buff_size=52428800)
        self.odom_sub = rospy.Subscriber(vins_odomtopic, Odometry, self.odom_callback,
                                         queue_size=1, buff_size=100)
        # path = "/home/flameycx/D435ipic/select_pic/frame0192.jpg"
        # image = cv2.imread(path)
        
        # segments = self.model(image)
        #cv2.imshow("ori",image)

        # cv2.imshow("ori",image)
        # cv2.waitKey(0)


        # image = cv2.imread("/media/data/yolo_data/dooraplate/split/val/319/2127.jpg")
        # resized_image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_LINEAR)
        # predict_result = self.model_num.predict(resized_image)
        
        # labels = predict_result[0].names  #名字
        # predictt_label = predict_result[0].probs.top1  #概率最高的索引
        # predict_probility = predict_result[0].probs.top1conf.cpu()  #概率值
        
        # print(f" {labels[predictt_label]} 概率为{predict_probility * 100:.2f}%")




        #print(rea[0])
        # output publishers
        self.position_pub = rospy.Publisher(
            pub_topic,  BoundingBoxes, queue_size=1)

        self.door_pub = rospy.Publisher(
            door_pub_topic,  String, queue_size=1)
        
        self.image_pub = rospy.Publisher(
            '/yolov8/detection_image',  Image, queue_size=1)
        
        self.pose_pub = rospy.Publisher('/roadsign_pose', PoseStamped, queue_size=10)
    
        # if no image messages
        while (not self.getImageStatus):
            rospy.loginfo("waiting for image.")
            rospy.sleep(0.5)



    def image_callback(self, image):

        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image.header
        self.boundingBoxes.image_header = image.header
        self.getImageStatus = True
        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1)

        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

        results = self.model(self.color_image, show=False, conf=0.3,verbose=False)

       # self.dectshow(results, image.height, image.width)
  
        # 没有检测到门牌时跳过后续
        if not results[0].boxes:
            # cv2.imshow('empty', self.color_image)  
            return  

        self.corner_detect(results)

        cv2.waitKey(3)

    def odom_callback(self,msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        
        orientation_list = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]  
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(orientation_list)
        self.odom_yaw = yaw*180/3.14
        self.okk=False
        self.vinsinit = True
        



    def corner_detect(self, results):

        self.frame = results[0].plot()

        for result in results[0].boxes:
            xmin = np.int64(result.xyxy[0][0].item())
            ymin = np.int64(result.xyxy[0][1].item())
            xmax = np.int64(result.xyxy[0][2].item())
            ymax = np.int64(result.xyxy[0][3].item())
            #cv2.rectangle(self.image, (xmin, ymin), (xmax, ymax), (0, 255, 0))
            #print("seg_points:",(xmin,ymin),(xmax,ymax))

            #mask_raw = segments[0].masks[0].cpu().data.numpy().transpose(1,2,0) 
            

            # 读取图像  
            image_gar = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2GRAY)  # 以灰度模式读取图像  


            cropped_image = image_gar[ymin-3:ymax+3, xmin-3:xmax+3] 

            cut_img = self.color_image[ymin-3:ymax+3, xmin-3:xmax+3]
            save_image = self.color_image[ymin:ymax, xmin:xmax]


            # 使用Shi-Tomasi算法检测角点  
            # maxCorners：最大角点数  
            # qualityLevel：最小质量因子，低于此值的角点将被拒绝  
            # minDistance：角点之间的最小欧氏距离  
            corners = cv2.goodFeaturesToTrack(cropped_image, maxCorners=100, qualityLevel=0.2, minDistance=5)  
            
            # corners将是一个包含角点坐标的numpy数组，每个角点是一个(x, y)坐标  

            if corners is None :
                return
            else:
                if corners.shape[0] <4 :
                    return
            
            if self.saveimg :
                output_dir = "/media/data/yolo_data/doorplate/image2"
                filename = f"{self.img_i}.jpg"
                output_path = os.path.join(output_dir, filename)

                cv2.imwrite(output_path, save_image)

                self.img_i = self.img_i + 1

            corners = np.int0(corners)  
            

            ori_corners = []   
            for sublist in corners:  
                # 假设每个子列表只有一个子列表，并且它包含两个元素  
                x, y = sublist[0]  # 取出(x, y)  
                # 计算新的x和y值  
                new_x = x + xmin-3
                new_y = y + ymin-3 
                # 将新的(x, y)作为一个新的子列表添加到processed_list中  
                # 注意：这里保持了原始列表的结构，即每个元素都是一个包含两个元素的子列表  
                ori_corners.append([[new_x, new_y]])  
            
            ori_corners = np.array(ori_corners)
            # 打印处理后的列表  
            

            ori_corners_list = []
            for i in ori_corners:  
                x, y = i.ravel()  
                ori_corners_list.append([x,y])
                cv2.circle(self.frame, (x, y), 1, (255,255,255), -1)  
                cv2.circle(cut_img, (x-xmin+3, y-ymin+3), 1, (255,255,255), -1)
            
            if self.okk:
                pose=[-1.2,0.3,1.6]
                quat=[0.833,0.015,-0.475,-0.282]
               # self.publish_pnp(pose, quat)
                       

            # mask_raw = []
            # for result in segments[0].masks.xy:
            #     #print(result)
            #     for xy in result:
            #         x = np.int64(xy[0].item())
            #         y = np.int64(xy[1].item())
            #         ori_corners_list.append([x,y])
            #         cv2.circle(image, (x,y), 1, (255, 255, 255), -1)
            
            points = np.array(ori_corners_list) 
            # points = np.array(mask_raw) 
            # print(points)
            #zeroimg = 255 * np.ones((image.shape), dtype=np.uint8)
            #zeroimg = cv2.fillPoly(zeroimg, [points], color=(0, 0, 255))
            
            #cv2.drawContours(zeroimg, [points], -1, (0, 0, 255), cv2.FILLED)
            #alpha = 1 
            #beta = 0.4   
            #gamma = 0  # 简单的加权和，不考虑gamma  
            #image = cv2.addWeighted(image, alpha, zeroimg, beta, gamma) 
            #找出离读取框最近的边缘点
            # 以端点坐标 (xmin, ymin, xmax, ymax)绘制四条边界框直线
            line1 = (xmin, ymin, xmax, ymin)
            line2 = (xmin, ymin, xmin, ymax)
            line3 = (xmax, ymin, xmax, ymax)
            line4 = (xmin, ymax, xmax, ymax)

            lines = [line1, line2, line3, line4]

            def get_line_length(line):
                x1, y1, x2, y2 = line
                return np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            length_x = get_line_length(line1) 
            length_y = get_line_length(line2) 
            #print("line1:", length_x)

            area = length_x*length_y
            #print("area:", area)
                

            # 存储距离小于阈值的点
            filtered_points = []

            # 根据检测框的面积定义固定的阈值
            if area > 1000:
                threshold_distance = 10  # 假设阈值为 20 像素
            else:
                threshold_distance = 6

            
            for point in points:
                px, py = point
                #中间部分去除
                if xmin+8 < px <xmax-8:
                    continue
                    
                for line in lines:
                    x1, y1, x2, y2 = line
            
                    # 计算点到直线的距离
                    distance = np.abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
            
                    # 如果距离小于阈值且到顶点的最小距离小于阈值，将点加入筛选集合
                    if distance < threshold_distance:
                        #cv2.circle(image, (px, py), 1, (255, 0, 0), -1)
                        filtered_points.append((px,py))
                        
                        break  # 如果已经加入筛选集合，可以提前结束内层循环

                    # #右下角的点要求特殊
                    # if (xmin+xmax)/2 < px < xmax and (ymin+ymax)/2 < py < ymax :
                    #     if distance < threshold_distance_1:
                    #         cv2.circle(image, (px, py), 1, (255, 0, 0), -1)
                    #         filtered_points.append((px,py))
                    #         break  # 如果已经加入筛选集合，可以提前结束内层循环
                    # else: 
                    #     if distance < threshold_distance:
                    #         cv2.circle(image, (px, py), 1, (255, 0, 0), -1)
                    #         filtered_points.append((px,py))
                    #         break  # 如果已经加入筛选集合，可以提前结束内层循环

            #print("all points:",filtered_points)
            
            #距离筛选
            if len(filtered_points) >= 4 :
                last_point = [] 
                
                all_dis = []
                filter_dis = []  
                for x,y in filtered_points:
                    dis = (xmin - x)**2 + (ymin-y)**2
                    filter_dis.append(dis)

                all_dis.append(filter_dis[min(enumerate(filter_dis), key=lambda x: x[1])[0]])
                min_value_index0 = min(enumerate(filter_dis), key=lambda x: x[1])[0]  
                last_point.append(filtered_points[min_value_index0])

                filter_dis = []  
                for x,y in filtered_points:
                    dis = (xmax - x)**2 + (ymin-y)**2
                    filter_dis.append(dis)
                all_dis.append(filter_dis[min(enumerate(filter_dis), key=lambda x: x[1])[0]])
                min_value_index1 = min(enumerate(filter_dis), key=lambda x: x[1])[0]  
                last_point.append(filtered_points[min_value_index1])

                filter_dis = []  
                for x,y in filtered_points:
                    dis = (xmax - x)**2 + (ymax-y)**2
                    filter_dis.append(dis)
                all_dis.append(filter_dis[min(enumerate(filter_dis), key=lambda x: x[1])[0]])
                min_value_index2 = min(enumerate(filter_dis), key=lambda x: x[1])[0]  
                last_point.append(filtered_points[min_value_index2])

                filter_dis = []  
                for x,y in filtered_points:
                    dis = (xmin - x)**2 + (ymax-y)**2
                    filter_dis.append(dis)
                all_dis.append(filter_dis[min(enumerate(filter_dis), key=lambda x: x[1])[0]])
                min_value_index3 = min(enumerate(filter_dis), key=lambda x: x[1])[0]  
                last_point.append(filtered_points[min_value_index3])

                #print("allmindis",min(enumerate(all_dis), key=lambda x: x[1])[0])
                if min(enumerate(all_dis), key=lambda x: x[1])[0] == 0 or min(enumerate(all_dis), key=lambda x: x[1])[0] == 2:
                    if min_value_index0>min_value_index2 :
                        filtered_points.pop(min_value_index0)
                        filtered_points.pop(min_value_index2)
                    else :
                        filtered_points.pop(min_value_index2)
                        filtered_points.pop(min_value_index0)
                    filter_dis = []
                    for x,y in filtered_points:
                        x1 = xmax
                        y1 = ymin
                        x2 = xmax
                        y2 = ymax
                        distance = np.abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2) + (xmax - x)**2 + (ymin-y)**2
                        filter_dis.append(distance)
                        
                    last_point[1] = filtered_points[min(enumerate(filter_dis), key=lambda x: x[1])[0]]

                    filter_dis = []
                    for x,y in filtered_points:
                        x1 = xmin
                        y1 = ymin
                        x2 = xmin
                        y2 = ymax
                        distance = np.abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2) + (xmin - x)**2 + (ymax-y)**2
                        filter_dis.append(distance)
                    last_point[3] = filtered_points[min(enumerate(filter_dis), key=lambda x: x[1])[0]]

                else:
                    filter_dis = []
                    if min_value_index1 > min_value_index3:
                        filtered_points.pop(min_value_index1)
                        filtered_points.pop(min_value_index3)
                    else:
                        filtered_points.pop(min_value_index3)
                        filtered_points.pop(min_value_index1)
                    #print(filtered_points)
                    for x,y in filtered_points:
                        x1 = xmax
                        y1 = ymin
                        x2 = xmax
                        y2 = ymax
                        distance = np.abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2) + (xmax - x)**2 + (ymax-y)**2
                        filter_dis.append(distance)
                    last_point[2] = filtered_points[min(enumerate(filter_dis), key=lambda x: x[1])[0]]

                    filter_dis = []
                    for x,y in filtered_points:
                        x1 = xmin
                        y1 = ymin
                        x2 = xmin
                        y2 = ymax
                        distance = np.abs((y2-y1)* x - (x2-x1)*y + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2) + (xmin - x)**2 + (ymin-y)**2
                        filter_dis.append(distance)
                    last_point[0] = filtered_points[min(enumerate(filter_dis), key=lambda x: x[1])[0]]



                filtered_points = last_point
                for x,y in filtered_points:
                    cv2.circle(self.frame, (x, y), 1, (255, 0, 0), -1)
                    cv2.circle(cut_img, (x-xmin+3, y-ymin+3), 1, (255, 0, 0), -1)
                    
                #print(filtered_points)
                vertex_points = [] #存储检测出的四个角点坐标
                vertex_points = filtered_points

                
                # cv2.imshow("ori",image)
                # cv2.waitKey(0)


                # 将筛选后的点集合转换为 numpy 数组
                # filtered_points = np.array(filtered_points)

                # # 寻找这些点的凸包
                # hull = cv2.convexHull(filtered_points)

                # # 轮廓近似
                # epsilon = 0.08 * cv2.arcLength(hull, True)
                # approx = cv2.approxPolyDP(hull, epsilon, True)

                # # 筛选出四边形的顶点
                # if len(approx) == 4:
                # # 在这里可以进一步添加对四个角点进行平行四边形几何特征筛选，不符合的舍弃掉进入下一张图片
                #     for point in approx:
                #         x, y = point[0]
                #         cv2.circle(image, (x,y), 1, (0, 0, 255), -1)
                #         vertex_points.append((x, y))

                #cenpoint = cross_point([xmin,ymin,xmax,ymax],[xmax,ymin,xmin,ymax])
                
                #cv2.circle(self.frame,(int(cenpoint[0]),int(cenpoint[1])), 1, (0, 0, 255), -1)
                

                thita1 = abs(calculate_angle_between_vectors(vertex_points[0],vertex_points[1],vertex_points[3],vertex_points[2]))
                thita2 = abs(calculate_angle_between_vectors(vertex_points[3],vertex_points[0],vertex_points[2],vertex_points[1]))
                #print("thita1",thita1,"thita2",thita2)

                resized_image = cv2.resize(save_image, (96, 96), interpolation=cv2.INTER_LINEAR)
                num_results = self.model_num(resized_image,verbose=False)
                
                labels = num_results[0].names  #名字
                predictt_label = num_results[0].probs.top1  #概率最高的索引
                predict_probility = num_results[0].probs.top1conf.cpu()  #概率值
                doorplate_num = labels[predictt_label]
                
                if not self.vinsinit:
                    if predict_probility >= 0.2 :
                    
                        self.doornum_flag = True 
                        self.use_doorplate =  "321"
                       

                else:
                    if self.odom_x < 0 and self.odom_yaw < 0:
                        self.doornum_flag = True 
                        self.use_doorplate =  "321"
                    # elif self.odom_x > 10 and self.odom_x < 13 and self.odom_yaw < 0:
                    #     self.doornum_flag = True 
                    #     self.use_doorplate =  "319"
                    elif self.odom_x > 10 and self.odom_x < 16 and self.odom_yaw < 0:
                        self.doornum_flag = True 
                        self.use_doorplate =  "317"

                    elif self.odom_x > 25 and self.odom_x < 29.79 and self.odom_yaw < 0:
                        self.doornum_flag = True 
                        self.use_doorplate =  "315"

                    elif self.odom_x > 30 and self.odom_x < 32.17 and self.odom_yaw < 0:
                        self.doornum_flag = True 
                        self.use_doorplate =  "313"

                    elif self.odom_x > 33 and self.odom_x < 41.5 and self.odom_yaw < 0 :
                        self.doornum_flag = True 
                        self.use_doorplate =  "311"

                    elif self.odom_x > 41.5  and self.odom_yaw < 0:
                        self.doornum_flag = True 
                        self.use_doorplate =  "309"

                    elif self.odom_x > 45  and self.odom_yaw > 0:
                        self.doornum_flag = True 
                        self.use_doorplate =  "310"

                    elif self.odom_x > 37.5 and self.odom_x < 40 and self.odom_yaw > 0:
                        self.doornum_flag = True 
                        self.use_doorplate =  "312"

                    elif self.odom_x > 32 and self.odom_x < 35 and self.odom_yaw > 0:
                        self.doornum_flag = True 
                        self.use_doorplate =  "314"

                    elif self.odom_x > 16 and self.odom_x < 20 and self.odom_yaw > 0:
                        self.doornum_flag = True 
                        self.use_doorplate =  "318"

                    elif self.odom_x > 8 and self.odom_x < 12 and self.odom_yaw > 0:
                        self.doornum_flag = True 
                        self.use_doorplate =  "322"

                    elif self.odom_x > 0 and self.odom_x < 5 and self.odom_yaw > 0:
                        self.doornum_flag = True 
                        self.use_doorplate =  "324"

                    else :
                        self.doornum_flag = False
                         

                if thita1<5 and thita2<5:
                   
                    #print("get corners")
                    # 重新按顺时针排列角点坐标，计算重心
                    # centroid = np.mean(vertex_points, axis=0)

                    # # 找到最小的坐标点作为起点(可能存在集合为空报错的情况)
                    # # start_point = min(vertex_points)

                    # # 找到距离原点最近的点作为起点
                    # def distance_to_origin(point):
                    #     return np.sum(point**2)  # 欧几里得距离的平方

                    # start_point_index = np.argmin(np.apply_along_axis(distance_to_origin, 1, vertex_points))
                    # start_point = vertex_points[start_point_index]


                    # # 计算每个顶点相对于重心的极角
                    # def calculate_angle(point):
                    #     angle = np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
                    #     return angle
                    
                    # #把角点按顺时针顺序排列但是起点坐标随机
                    # sorted_vertex_points = sorted(vertex_points, key=calculate_angle)
                    
                    # #找出左上方角点坐标在点集中的索引位置
                    # start_index = sorted_vertex_points.index(start_point)

                    # # 调整顺序以从起点开始顺时针排序
                    # sorted_vertex_points = sorted_vertex_points[start_index:] + sorted_vertex_points[:start_index]

                    sorted_vertex_points = filtered_points
                    #sorted_vertex_points.append(cross_point([xmin,ymin,xmax,ymax],[xmax,ymin,xmin,ymax]))
                    # print("Vertex points:", vertex_points)
                    #print("sorted points:", sorted_vertex_points) 

                    
                        
                    
                    if self.doornum_flag :
                        
                        index = self.object.index(self.use_doorplate)  
                        # 使用找到的索引从坐标数组中取出相应的坐标  
                        x_doorplate, y_doorplate = self.coordinates[index] 
                        print("get doorplat :",self.use_doorplate)

                        num = int(self.use_doorplate)  
                        # 使用模运算符判断最后一位是奇数还是偶数  
                        if num % 10 % 2 == 0:  
                            object_points = np.array([[x_doorplate-0.1625/2, y_doorplate, 2.141+0.031],  
                                        [x_doorplate+0.1625/2, y_doorplate, 2.141+0.031],  
                                        [x_doorplate+0.1625/2, y_doorplate, 2.141-0.031],  
                                        [x_doorplate-0.1625/2, y_doorplate, 2.141-0.031]
                                        ], dtype=np.float32)
                        else:  
                            object_points = np.array([[x_doorplate+0.1625/2, y_doorplate, 2.141+0.031],  
                                        [x_doorplate-0.1625/2, y_doorplate, 2.141+0.031],  
                                        [x_doorplate-0.1625/2, y_doorplate, 2.141-0.031],  
                                        [x_doorplate+0.1625/2, y_doorplate, 2.141-0.031]
                                        ], dtype=np.float32)  
                        
                        
                        
                        # 假设对应的图像坐标（通常是经过相机内参矫正和畸变矫正的）  
                        image_points = np.array(sorted_vertex_points, dtype=np.float32)  
                        
                        # 假设相机的内参矩阵（焦距fx, fy, 光心cx, cy）  
                        # camera_matrix = np.array([[890.9319704879239, 0, 661.4530888480895],  
                        #                         [0, 889.2287448089522, 376.4407234933706],  
                        #                         [0, 0, 1]], dtype=np.float32)  
                        camera_matrix = np.array([[581.6665241614161, 0, 331.22269265482754],  
                                                [0, 579.8579509101475, 248.70801108943016],  
                                                [0, 0, 1]], dtype=np.float32)  
                        
                        # 畸变  
                        dist_coeffs = np.array([0.1267126636566763, -0.28632741474982865, -0.0020898900401240916, 0.00010039599818046335])
                        
                        # 使用solvePnP求解位姿  
                        retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)  

                        # camera2imu = np.array([[0.01217497,-0.03002283,0.99947506],
                        # [-0.99992494,-0.00173672,0.01212829],
                        # [0.00137168,-0.9995477,-0.03004172]])


                        camera2imu = np.array([[ -1.6280327243022086e-02, 2.2078592565333732e-02,9.9962367253641737e-01], 
                        [-9.9986000496366900e-01,3.5029036128054747e-03, -1.6361544558244678e-02,],
                        [-3.8628252500148563e-03, -9.9975010148367260e-01,2.2018473254848070e-02 ]])
                        
                        rotation_matrix, _ = cv2.Rodrigues(rvec)  
                        
                        pose = -rotation_matrix.T.dot(tvec)
                            
                        #旋转矩阵
                        
                        rot_imu = R.from_matrix(rotation_matrix.T.dot(camera2imu.T))
                        rot = R.from_matrix(rotation_matrix.T)
                        euler_angles = rot_imu.as_euler('xyz', degrees=True)
                        
                        if pose[2] < 1.8 and pose[2] > 1.2 and pose[1] > -1.255 and pose[1] < 1.255 and self.judge(self.use_doorplate,pose):
                            # 使用SciPy转换旋转矩阵为四元数
                            quat = rot.as_quat()
                            print("roll pitch yall", euler_angles)
                            print("Translation Vector:\n", pose)
                            #self.publish_pnp(pose, quat)
                            # if self.visualize :
                            #     cv2.imshow('YOLOv8', cut_img)
                        else :
                            print("Not good pose")
                    else :
                        print("Not sure of the doorplate number")
                else :
                    print("No correct points")
            else:
                print("No enough points")

            
            #cv2.imshow("ori",zeroimg)
            # cv2.imshow("ori",self.image)
            # cv2.waitKey(0)

    def publish_pnp(self, pose, quat):

        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = self.camera_frame  
        pose_msg.pose.position.x = pose[0]
        pose_msg.pose.position.y = pose[1]
        pose_msg.pose.position.z = pose[2]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        #self.door_pub.publish(self.use_doorplate)
        self.pose_pub.publish(pose_msg)

    def dectshow(self, results, height, width):

        self.frame = results[0].plot()
        #print(str(results[0].speed['inference']))
        fps = 1000.0/ results[0].speed['inference']
        cv2.putText(self.frame, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        for result in results[0].boxes:
            boundingBox = BoundingBox()
            boundingBox.xmin = np.int64(result.xyxy[0][0].item())
            boundingBox.ymin = np.int64(result.xyxy[0][1].item())
            boundingBox.xmax = np.int64(result.xyxy[0][2].item())
            boundingBox.ymax = np.int64(result.xyxy[0][3].item())
            boundingBox.Class = results[0].names[result.cls.item()]
            boundingBox.probability = result.conf.item()
            self.boundingBoxes.bounding_boxes.append(boundingBox)
        self.position_pub.publish(self.boundingBoxes)
        self.publish_image(self.frame, height, width)

    def judge(self,doornum,pose):
        if doornum == "321" and pose[0] < 0 and pose[1] > 0 and pose[2] > 1.5:
            return True
        # elif doornum == "319" and pose[0] < 13 and pose[1] > 0:
        #     return True
        elif doornum == "317" and pose[0] > 13 and pose[0] < 16 and pose[1] > 0:
            return True
        
        elif doornum == "315" and pose[0] < 29.79 and pose[1] > 0:
            return True
        
        elif doornum == "313" and pose[0] < 32.17 and pose[1] > 0:
            return True
        
        elif doornum == "311" and pose[0] < 37 and pose[1] > 0:
            return True
        
        elif doornum == "309" and pose[0] < 45 and pose[1] > 0:
            return True
        
        elif doornum == "310" and pose[0] > 45 and pose[1] < 0:
            return True
        
        elif doornum == "312" and pose[0] > 37 and pose[1] < 0:
            return True
        
        elif doornum == "314" and pose[0] > 32 and pose[1] > -0.5 and pose[1]<0:
            return True
        
        elif doornum == "318" and pose[0] > 16 and pose[1] < 0:
            return True
        
        elif doornum == "322" and pose[0] > 8 and pose[1] < 0:
            return True
        
        elif doornum == "324" and pose[0] > 0 and pose[1] < 0:
            return True
        
        else :
            return False
    def publish_image(self, imgdata, height, width):
        image_temp = Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = self.camera_frame
        image_temp.height = height
        image_temp.width = width
        image_temp.encoding = 'bgr8'
        image_temp.data = np.array(imgdata).tobytes()
        image_temp.header = header
        image_temp.step = width * 3
        self.image_pub.publish(image_temp)

  
            

def main():
    rospy.init_node('yolov8_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()


if __name__ == "__main__":

    main()
