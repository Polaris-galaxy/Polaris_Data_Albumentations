#!/usr/bin/env python3
"""
Ubuntu USB摄像头 Deformable DETR 目标检测
适用于USB摄像头和CSI摄像头
"""

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import argparse
import time
import os
import sys
import subprocess
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class USBCameraDeformableDETR:
    def __init__(self, model_path, confidence_threshold=0.5, camera_resolution=(1280, 720)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        self.confidence_threshold = confidence_threshold
        self.camera_resolution = camera_resolution
        
        # 初始化摄像头状态
        self.camera_initialized = False
        self.cap = None
        
        try:
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 加载模型
            logger.info("加载模型...")
            if model_path.endswith('.pth') or model_path.endswith('.pt'):
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # 处理不同的模型保存格式
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                    
                    # 构建模型
                    self.model = self._build_model_from_checkpoint(state_dict)
                    
                    # 加载权重
                    self.model.load_state_dict(state_dict, strict=False)
                else:
                    # 如果checkpoint本身就是模型
                    self.model = checkpoint
                
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"模型加载成功: {model_path}")
            else:
                raise ValueError("不支持的模型格式")
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            sys.exit(1)
        
        # COCO 数据集类别名称
        self.CLASSES = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # 图像预处理
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 检测结果记录
        self.detection_log = []
    
    def _build_model_from_checkpoint(self, state_dict):
        """根据状态字典构建模型结构"""
        # 这是一个简化的模型构建，您可能需要根据实际模型结构进行调整
        try:
            # 尝试导入Deformable DETR模型
            sys.path.append('/path/to/Deformable-DETR')  # 添加您的Deformable DETR路径
            
            from models.deformable_detr import DeformableDETR
            from models.backbone import Backbone
            
            # 创建模型实例（参数需要根据您的模型调整）
            backbone = Backbone('resnet50', train_backbone=True, return_interm_layers=True)
            model = DeformableDETR(
                backbone,
                num_classes=91,  # COCO数据集有91个类别（包括背景）
                num_queries=300,
            )
            return model
        except ImportError:
            logger.warning("无法导入Deformable DETR，使用模拟模型")
            return self._create_dummy_model()
    
    def _create_dummy_model(self):
        """创建模拟模型用于测试"""
        import torch.nn as nn
        
        class DummyModel(nn.Module):
            def __init__(self):
                super(DummyModel, self).__init__()
                
            def forward(self, x):
                batch_size = x.shape[0]
                # 返回模拟的检测结果
                return {
                    'pred_logits': torch.randn(batch_size, 100, 91),  # 91 classes
                    'pred_boxes': torch.randn(batch_size, 100, 4)
                }
        
        return DummyModel()
    
    def list_usb_cameras(self):
        """列出可用的USB摄像头"""
        logger.info("扫描USB摄像头...")
        
        # 方法1: 使用v4l2-ctl列出设备
        try:
            result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                cameras = []
                current_device = ""
                for line in lines:
                    if line.strip() and not line.startswith('\t'):
                        current_device = line.strip()
                    elif line.startswith('\t/dev/video'):
                        device_path = line.strip()
                        cameras.append({
                            'device': device_path,
                            'name': current_device,
                            'index': int(device_path.split('/dev/video')[-1])
                        })
                return cameras
        except Exception as e:
            logger.warning(f"无法使用v4l2-ctl: {e}")
        
        # 方法2: 检查/dev/video*设备
        cameras = []
        for i in range(10):
            device_path = f"/dev/video{i}"
            if os.path.exists(device_path):
                # 尝试打开设备获取信息
                cap = cv2.VideoCapture(device_path)
                if cap.isOpened():
                    cameras.append({
                        'device': device_path,
                        'name': f'Video Device {i}',
                        'index': i
                    })
                    cap.release()
        
        return cameras
    
    def init_usb_camera(self, camera_device="/dev/video0"):
        """初始化USB摄像头"""
        logger.info(f"初始化USB摄像头: {camera_device}")
        
        try:
            # 使用V4L2后端
            self.cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
            
            if not self.cap.isOpened():
                logger.error(f"无法打开摄像头: {camera_device}")
                return False
            
            # 设置摄像头参数
            width, height = self.camera_resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲延迟
            
            # 验证设置
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"摄像头设置: {actual_width}x{actual_height} @ {actual_fps}FPS")
            
            self.camera_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}")
            return False
    
    def init_csi_camera(self, sensor_id=0):
        """初始化CSI摄像头（适用于Jetson等设备）"""
        logger.info(f"初始化CSI摄像头，传感器ID: {sensor_id}")
        
        try:
            # CSI摄像头通常使用gstreamer管道
            if sensor_id == 0:
                pipeline = (
                    "nvarguscamerasrc ! "
                    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
                    "nvvidconv flip-method=0 ! "
                    "video/x-raw, width=1280, height=720, format=BGRx ! "
                    "videoconvert ! "
                    "video/x-raw, format=BGR ! "
                    "appsink"
                )
            else:
                pipeline = (
                    f"nvarguscamerasrc sensor-id={sensor_id} ! "
                    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
                    "nvvidconv flip-method=0 ! "
                    "video/x-raw, width=1280, height=720, format=BGRx ! "
                    "videoconvert ! "
                    "video/x-raw, format=BGR ! "
                    "appsink"
                )
            
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                logger.error("无法打开CSI摄像头")
                return False
            
            self.camera_initialized = True
            logger.info("CSI摄像头初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"CSI摄像头初始化失败: {e}")
            return False
    
    def preprocess_image(self, image):
        """预处理图像"""
        try:
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
            image_pil = Image.fromarray(image_rgb)
            image_tensor = self.transform(image_pil).unsqueeze(0)
            return image_tensor.to(self.device)
        except Exception as e:
            logger.error(f"图像预处理错误: {e}")
            return None
    
    def process_detections(self, outputs, frame):
        """处理检测结果"""
        try:
            h, w = frame.shape[:2]
            
            # 解析模型输出
            if isinstance(outputs, dict):
                pred_logits = outputs['pred_logits']
                pred_boxes = outputs['pred_boxes']
            elif hasattr(outputs, 'pred_logits') and hasattr(outputs, 'pred_boxes'):
                pred_logits = outputs.pred_logits
                pred_boxes = outputs.pred_boxes
            else:
                logger.warning("未知的输出格式")
                return [], frame
            
            # 应用softmax获取概率
            probas = pred_logits.softmax(-1)[0, :, :-1]  # 移除背景类
            keep = probas.max(-1).values > self.confidence_threshold
            
            detections = []
            
            if keep.sum() > 0:
                boxes = pred_boxes[0, keep]
                scores = probas[keep]
                class_indices = scores.argmax(-1)
                
                for box, score, class_idx in zip(boxes, scores, class_indices):
                    # 转换边界框坐标
                    x_center, y_center, width, height = box.tolist()
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    # 确保坐标在图像范围内
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(0, min(x2, w-1))
                    y2 = max(0, min(y2, h-1))
                    
                    # 跳过无效的边界框
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    class_id = class_idx.item()
                    confidence = score.max().item()
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': self.CLASSES[class_id] if class_id < len(self.CLASSES) else f'Class_{class_id}',
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
            
            return detections, frame
            
        except Exception as e:
            logger.error(f"处理检测结果错误: {e}")
            return [], frame
    
    def draw_detections(self, frame, detections):
        """在图像上绘制检测结果"""
        try:
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                # 为不同类别生成颜色
                color = self._get_color(detection['class_id'])
                
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # 绘制标签背景
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"绘制检测结果错误: {e}")
            return frame
    
    def _get_color(self, class_idx):
        """为不同类别生成不同颜色"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (255, 128, 0), (128, 255, 0), (0, 255, 128), (0, 128, 255), (128, 0, 255)
        ]
        return colors[class_idx % len(colors)]
    
    def log_detections(self, detections, timestamp):
        """记录检测结果"""
        for detection in detections:
            log_entry = {
                'timestamp': timestamp,
                'class_name': detection['class_name'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox']
            }
            self.detection_log.append(log_entry)
    
    def save_detection_log(self, filename="detection_log.json"):
        """保存检测日志"""
        import json
        try:
            with open(filename, 'w') as f:
                json.dump(self.detection_log, f, indent=2)
            logger.info(f"检测日志已保存: {filename}")
        except Exception as e:
            logger.error(f"保存检测日志失败: {e}")
    
    def run_detection(self, camera_type="usb", camera_device="/dev/video0", csi_sensor_id=0):
        """运行目标检测"""
        # 初始化摄像头
        if camera_type.lower() == "usb":
            if not self.init_usb_camera(camera_device):
                logger.error("USB摄像头初始化失败")
                return
        elif camera_type.lower() == "csi":
            if not self.init_csi_camera(csi_sensor_id):
                logger.error("CSI摄像头初始化失败")
                return
        else:
            logger.error(f"不支持的摄像头类型: {camera_type}")
            return
        
        logger.info("开始目标检测，按 'q' 退出，按 's' 保存图像，按 'l' 显示/隐藏检测标签")
        
        frame_count = 0
        start_time = time.time()
        show_detections = True
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("无法读取摄像头帧")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # 预处理和推理
                input_tensor = self.preprocess_image(frame)
                if input_tensor is None:
                    continue
                
                with torch.no_grad():
                    try:
                        outputs = self.model(input_tensor)
                    except Exception as e:
                        logger.error(f"推理错误: {e}")
                        continue
                
                # 处理检测结果
                detections, result_frame = self.process_detections(outputs, frame.copy())
                
                # 记录检测结果
                self.log_detections(detections, current_time)
                
                # 绘制检测结果
                if show_detections:
                    result_frame = self.draw_detections(result_frame, detections)
                
                # 计算并显示性能信息
                fps = frame_count / (current_time - start_time)
                cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Detections: {len(detections)}", (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Camera: {camera_type.upper()}", (10, 110),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示帧
                cv2.imshow('USB Camera - Deformable DETR', result_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存当前帧
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_{timestamp}.jpg"
                    cv2.imwrite(filename, result_frame)
                    logger.info(f"图像已保存: {filename}")
                elif key == ord('l'):
                    show_detections = not show_detections
                    logger.info(f"{'显示' if show_detections else '隐藏'}检测标签")
                
        except KeyboardInterrupt:
            logger.info("程序被用户中断")
        except Exception as e:
            logger.error(f"运行时错误: {e}")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # 保存检测日志
            self.save_detection_log()
            logger.info("摄像头已释放")

def main():
    parser = argparse.ArgumentParser(description='Ubuntu USB摄像头 Deformable DETR 目标检测')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--camera_type', type=str, default='usb', choices=['usb', 'csi'], 
                       help='摄像头类型: usb 或 csi (默认: usb)')
    parser.add_argument('--camera_device', type=str, default='/dev/video0', 
                       help='USB摄像头设备路径 (默认: /dev/video0)')
    parser.add_argument('--csi_sensor_id', type=int, default=0, 
                       help='CSI摄像头传感器ID (默认: 0)')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='置信度阈值 (默认: 0.5)')
    parser.add_argument('--resolution', type=str, default='1280x720', 
                       help='摄像头分辨率 (默认: 1280x720)')
    parser.add_argument('--list_cameras', action='store_true', 
                       help='列出可用的摄像头')
    
    args = parser.parse_args()
    
    # 解析分辨率
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        logger.warning(f"无效的分辨率格式: {args.resolution}，使用默认分辨率")
        resolution = (1280, 720)
    
    # 创建检测器实例
    detector = USBCameraDeformableDETR(
        model_path=args.model_path,
        confidence_threshold=args.confidence,
        camera_resolution=resolution
    )
    
    # 列出摄像头
    if args.list_cameras:
        cameras = detector.list_usb_cameras()
        if cameras:
            logger.info("可用的摄像头:")
            for cam in cameras:
                logger.info(f"  {cam['index']}: {cam['device']} - {cam['name']}")
        else:
            logger.info("未找到可用的摄像头")
        return
    
    # 运行检测
    detector.run_detection(
        camera_type=args.camera_type,
        camera_device=args.camera_device,
        csi_sensor_id=args.csi_sensor_id
    )

if __name__ == "__main__":
    main()

#python3 ubuntu_deformable_detr_camera.py --model_path /path/to/your/model.pth --camera_type usb --camera_device /dev/video0 --confidence 0.5