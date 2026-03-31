import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import argparse
import time
import os
import sys

# 尝试导入Deformable DETR相关模块
try:
    from models import build_model
    from util.misc import nested_tensor_from_tensor_list
    print("成功导入Deformable DETR模块")
except ImportError as e:
    print(f"导入Deformable DETR模块失败: {e}")
    print("请确保Deformable DETR代码库在Python路径中")

class DeformableDETRCamera:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        self.confidence_threshold = confidence_threshold
        
        try:
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 加载状态字典
            print("加载模型状态字典...")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 构建模型结构
            print("构建模型结构...")
            self.model = self.build_model(checkpoint)
            
            # 加载状态字典到模型
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
                
            # 处理状态字典的键名不匹配问题
            new_state_dict = {}
            for k, v in state_dict.items():
                # 移除可能的prefix
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v
            
            # 加载状态字典
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                print(f"警告: 缺失的键: {missing_keys}")
            if unexpected_keys:
                print(f"警告: 意外的键: {unexpected_keys}")
            
            self.model.to(self.device)
            self.model.eval()
            print(f"模型加载成功: {model_path}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
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
    
    def build_model(self, checkpoint):
        """根据检查点构建模型结构"""
        try:
            # 尝试从检查点获取模型参数
            if 'args' in checkpoint:
                args = checkpoint['args']
                print(f"从检查点获取模型参数: {args}")
            else:
                # 使用默认参数
                print("使用默认模型参数")
                class Args:
                    def __init__(self):
                        self.lr_backbone = 1e-5
                        self.masks = False
                        self.dilation = False
                        self.position_embedding = 'sine'
                        self.enc_layers = 6
                        self.dec_layers = 6
                        self.dim_feedforward = 1024
                        self.hidden_dim = 256
                        self.dropout = 0.1
                        self.nheads = 8
                        self.num_queries = 300
                        self.pre_norm = False
                        self.num_feature_levels = 4
                args = Args()
            
            # 构建模型
            model, criterion, postprocessors = build_model(args)
            return model
            
        except Exception as e:
            print(f"构建模型失败: {e}")
            # 如果构建失败，创建一个简单的模型结构用于测试
            print("创建测试模型结构...")
            return self.create_dummy_model()
    
    def create_dummy_model(self):
        """创建用于测试的简单模型结构"""
        import torch.nn as nn
        
        class DummyModel(nn.Module):
            def __init__(self):
                super(DummyModel, self).__init__()
                self.dummy_param = nn.Parameter(torch.randn(1))
                
            def forward(self, x):
                # 返回模拟的输出结构
                batch_size = x.shape[0]
                return {
                    'pred_logits': torch.randn(batch_size, 100, 92),  # 92 classes (91 + background)
                    'pred_boxes': torch.randn(batch_size, 100, 4)
                }
        
        return DummyModel()
    
    def preprocess_image(self, image):
        """预处理图像"""
        try:
            # 确保图像是BGR格式
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
            image_pil = Image.fromarray(image_rgb)
            image_tensor = self.transform(image_pil).unsqueeze(0)
            return image_tensor.to(self.device)
        except Exception as e:
            print(f"图像预处理错误: {e}")
            return None
    
    def process_outputs(self, outputs):
        """处理模型输出"""
        try:
            # 尝试不同的输出格式
            if isinstance(outputs, dict):
                # 如果是字典格式
                if 'pred_logits' in outputs and 'pred_boxes' in outputs:
                    return outputs['pred_logits'], outputs['pred_boxes']
            
            elif isinstance(outputs, (tuple, list)):
                # 如果是元组或列表格式
                if len(outputs) >= 2:
                    return outputs[0], outputs[1]
            
            # 尝试直接访问属性
            if hasattr(outputs, 'pred_logits') and hasattr(outputs, 'pred_boxes'):
                return outputs.pred_logits, outputs.pred_boxes
            
            print(f"未知的输出格式: {type(outputs)}")
            print(f"输出内容: {outputs}")
            return None, None
            
        except Exception as e:
            print(f"输出处理错误: {e}")
            return None, None
    
    def draw_detections(self, image, outputs):
        """在图像上绘制检测结果"""
        try:
            h, w = image.shape[:2]
            
            # 处理模型输出
            pred_logits, pred_boxes = self.process_outputs(outputs)
            if pred_logits is None or pred_boxes is None:
                cv2.putText(image, "No valid outputs", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return image
            
            # 获取检测结果
            probas = pred_logits.softmax(-1)[0, :, :-1]  # 移除背景类
            keep = probas.max(-1).values > self.confidence_threshold
            
            # 如果没有检测到任何目标，返回原图
            if keep.sum() == 0:
                cv2.putText(image, "No detections", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return image
            
            boxes = pred_boxes[0, keep]
            scores = probas[keep]
            class_indices = scores.argmax(-1)
            
            # 绘制边界框和标签
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
                
                # 绘制边界框
                color = self.get_color(class_idx.item())
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                class_id = class_idx.item()
                if class_id < len(self.CLASSES):
                    class_name = self.CLASSES[class_id]
                else:
                    class_name = f"Class_{class_id}"
                    
                label = f"{class_name}: {score.max().item():.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # 绘制标签背景
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return image
            
        except Exception as e:
            print(f"绘制检测结果错误: {e}")
            cv2.putText(image, f"Error: {str(e)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return image
    
    def get_color(self, class_idx):
        """为不同类别生成不同颜色"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
        ]
        return colors[class_idx % len(colors)]
    
    def run_camera(self, camera_id=0):
        """运行摄像头检测"""
        print(f"尝试打开摄像头 {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        # 设置摄像头分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            print("尝试其他摄像头ID...")
            # 尝试其他摄像头ID
            for i in range(1, 5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"成功打开摄像头 {i}")
                    camera_id = i
                    break
                cap.release()
            else:
                print("无法找到可用的摄像头")
                return
        
        print("摄像头打开成功")
        print("按 'q' 退出，按 's' 保存当前帧")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break
                
                frame_count += 1
                
                # 预处理和推理
                input_tensor = self.preprocess_image(frame)
                if input_tensor is None:
                    continue
                
                with torch.no_grad():
                    try:
                        outputs = self.model(input_tensor)
                    except Exception as e:
                        print(f"推理错误: {e}")
                        continue
                
                # 绘制检测结果
                result_frame = self.draw_detections(frame.copy(), outputs)
                
                # 计算并显示FPS
                current_time = time.time()
                fps = frame_count / (current_time - start_time)
                cv2.putText(result_frame, f"FPS: {fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Deformable DETR Camera', result_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存当前帧
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_{timestamp}.jpg"
                    cv2.imwrite(filename, result_frame)
                    print(f"图像已保存: {filename}")
                
        except KeyboardInterrupt:
            print("程序被用户中断")
        except Exception as e:
            print(f"运行时错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("摄像头已释放")

def test_model_loading(model_path):
    """测试模型加载"""
    print("测试模型加载...")
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"检查点类型: {type(checkpoint)}")
        print(f"检查点键: {checkpoint.keys() if isinstance(checkpoint, dict) else '不是字典'}")
        
        if isinstance(checkpoint, dict):
            for key in checkpoint.keys():
                print(f"  {key}: {type(checkpoint[key])}")
                if hasattr(checkpoint[key], 'shape'):
                    print(f"    形状: {checkpoint[key].shape}")
    except Exception as e:
        print(f"测试模型加载失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='Deformable DETR 摄像头检测')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型权重路径')
    parser.add_argument('--camera_id', type=int, default=0, help='摄像头ID (默认: 0)')
    parser.add_argument('--confidence', type=float, default=0.5, help='置信度阈值 (默认: 0.5)')
    parser.add_argument('--test_model', action='store_true', help='测试模型加载')
    
    args = parser.parse_args()
    
    if args.test_model:
        test_model_loading(args.model_path)
        return
    
    # 创建检测器实例
    detector = DeformableDETRCamera(args.model_path, args.confidence)
    
    # 运行摄像头检测
    detector.run_camera(args.camera_id)

if __name__ == "__main__":
    main()
# python detr_opencv_camera_detect_OpenCV摄像头检测.py --model_path C:\Users\31447\Deformable-DETR\结果文件\final_model.pth --camera_id 0 --confidence 0.5