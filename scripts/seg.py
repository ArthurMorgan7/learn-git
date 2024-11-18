#!/usr/bin/env python3
# 指定脚本使用的Python解释器

import cv_bridge # 导入cv_bridge模块，用于ROS图像消息和OpenCV图像格式之间的转换
import numpy as np
import roslib.packages
import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult

class TrackerNode:
    def __init__(self):
        # 初始化节点的参数
        # 指定接收图像数据的ROS话题
        self.input_topic = rospy.get_param("~input_topic", "image_raw")
        # 将检测结果发布到的ROS话题
        self.result_topic = rospy.get_param("~result_topic", "yolo_result")
        # 结果图像发布的ROS话题名称
        self.result_image_topic = rospy.get_param("~result_image_topic", "yolo_image")

        # 指定要使用的YOLO模型
        yolo_model = rospy.get_param("~yolo_model", "yolov8n.pt")
        # 置信度阈值，默认值为0.25。用于过滤检测结果的置信度阈值，低于该值的结果会被丢弃。
        self.conf_thres = rospy.get_param("~conf_thres", 0.25) 
        # IoU（交并比）阈值，默认值为0.45。表示两个边界框之间的最小IoU阈值，用于非极大值抑制（NMS）。
        self.iou_thres = rospy.get_param("~iou_thres", 0.45)
        # 最大检测数，默认值为300。指定每张图像上允许的最大检测数量。
        self.max_det = rospy.get_param("~max_det", 300)
        # 需要检测的类别，默认为None。可以指定要检测的特定类别，如果为None，则检测所有类别。
        self.classes = rospy.get_param("~classes", None)
        # 跟踪器的配置文件路径或名称，默认值为"bytetrack.yaml"。用于指定目标跟踪器的配置。
        self.tracker = rospy.get_param("~tracker", "bytetrack.yaml")
        # 设备类型，默认为None。指定模型的运行显卡设备，例如"cpu"或"cuda:0"等。
        self.device = rospy.get_param("~device", None)

        # result_conf：是否在结果图像中显示置信度，默认为True。控制是否在输出图像中显示检测框的置信度。
        # result_line_width：结果图像中检测框的线宽，默认为None。指定在结果图像中绘制检测框的线宽。
        # result_font_size：结果图像中文本的字体大小，默认为None。指定在结果图像中显示的文本的字体大小。
        # result_font：结果图像中文本的字体名称，默认为"Arial.ttf"。指定在结果图像中显示的文本的字体。
        # result_labels：是否在结果图像中显示标签，默认为True。控制是否在输出图像中显示类别标签。
        # result_boxes：是否在结果图像中显示边界框，默认为True。控制是否在输出图像中显示检测框。
        self.result_conf = rospy.get_param("~result_conf", True)
        self.result_line_width = rospy.get_param("~result_line_width", None)
        self.result_font_size = rospy.get_param("~result_font_size", None)
        self.result_font = rospy.get_param("~result_font", "Arial.ttf")
        self.result_labels = rospy.get_param("~result_labels", True)
        self.result_boxes = rospy.get_param("~result_boxes", True)

        # 加载YOLO模型
        path = roslib.packages.get_pkg_dir("ultralytics_ros")  # 获取ultralytics_ros软件包的路径
        self.model = YOLO(f"{path}/models/{yolo_model}")  # 加载YOLO模型
        self.model.fuse()  # 加载模型权重

        # 创建ROS订阅器和发布器
        self.sub = rospy.Subscriber(
            self.input_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.results_pub = rospy.Publisher(self.result_topic, YoloResult, queue_size=1)
        self.result_image_pub = rospy.Publisher(self.result_image_topic, Image, queue_size=1)
        self.bridge = cv_bridge.CvBridge()

        # 检查是否使用分割模型
        self.use_segmentation = yolo_model.endswith("-seg.pt")

    # 图像回调函数，用于接收传感器图像消息
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # 使用YOLO模型进行目标跟踪
        results = self.model.track(
            source=cv_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
            retina_masks=True,
        )

        # 如果有检测结果
        if results is not None: 
            yolo_result_image_msg = Image()  # 创建图像消息
            yolo_result_image_msg.header = msg.header  # 设置图像消息头部
            yolo_result_image_msg = self.create_result_image(results)  # 创建结果图像消息
            self.result_image_pub.publish(yolo_result_image_msg)  # 发布结果图像消息

            # YoloResult消息类型包含detections和掩膜图像
            # header
            # detections: 
            #     heade
            #     detections: "<array type: vision_msgs/Detection2D, length: 5>"
            # masks: "<array type: sensor_msgs/Image, length: 5>"
            yolo_result_msg = YoloResult()  # 创建YoloResult消息
            yolo_result_msg.header = msg.header  # 设置消息头部
            yolo_result_msg.detections = self.create_detections_array(results)  # 创建检测结果的消息数组
            if self.use_segmentation:  # 如果使用分割模型
                yolo_result_msg.masks = self.create_segmentation_masks(results)  # 创建分割掩模消息
            self.results_pub.publish(yolo_result_msg)  # 发布检测结果消息

    # 创建检测结果数组消息
    def create_detections_array(self, results):
        detections_msg = Detection2DArray()
        # 获取坐标、类别id和相似度 [0]是因为结果是一个列表，在处理多张图片时用
        # 边界框的坐标格式为 [x, y, w, h]，其中 x 和 y 是边界框左上角的坐标，w 和 h 分别表示边界框的宽度和高度
        bounding_box = results[0].boxes.xywh 
        classes = results[0].boxes.cls
        confidence_score = results[0].boxes.conf
        for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
            detection = Detection2D()
            detection.bbox.center.x = float(bbox[0])
            detection.bbox.center.y = float(bbox[1])
            detection.bbox.size_x = float(bbox[2])
            detection.bbox.size_y = float(bbox[3])
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(cls)
            hypothesis.score = float(conf)
            detection.results.append(hypothesis)
            detections_msg.detections.append(detection)
        return detections_msg

    # 创建结果图像消息
    def create_result_image(self, results):
        # 在图像上绘制分割结果
        plotted_image = results[0].plot(
            conf=self.result_conf,
            line_width=self.result_line_width,
            font_size=self.result_font_size,
            font=self.result_font,
            labels=self.result_labels,
            boxes=self.result_boxes,
        )
        result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")
        return result_image_msg

    # 创建分割掩模消息
    def create_segmentation_masks(self, results):
        masks_msg = []
        for result in results:
            if hasattr(result, "masks") and result.masks is not None:
                # 遍历 result.masks 中的每个分割掩模
                for mask_tensor in result.masks:
                    # 将 PyTorch 张量（tensor）转换为 NumPy 数组（numpy array）
                    mask_numpy = (
                        np.squeeze(mask_tensor.data.to("cpu").detach().numpy()).astype(
                            np.uint8
                        )
                        * 255
                    )
                    # 将 NumPy 数组转换为 ROS 图像消息
                    mask_image_msg = self.bridge.cv2_to_imgmsg(
                        mask_numpy, encoding="mono8"
                    )
                    masks_msg.append(mask_image_msg)
        return masks_msg


if __name__ == "__main__":
    rospy.init_node("tracker_node")  # 初始化ROS节点
    node = TrackerNode()  # 实例化TrackerNode类
    rospy.spin()  # 进入ROS事件循环
