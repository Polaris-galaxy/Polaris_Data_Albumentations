import albumentations as A
import cv2
import numpy as np
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import math

class YOLOAugmentor:
    def __init__(self, image_size=640):
        self.image_size = image_size
        
        # 基础增强管道
        self.base_transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
        ], bbox_params=A.BboxParams(
            format='yolo',
            min_area=1,
            min_visibility=0.1,
            label_fields=['class_labels']
        ))
        
        # 几何增强管道
        self.geometric_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(0.1, 0.1),
                rotate=(-15, 15),
                shear=(-5, 5),
                p=0.5
            ),
        ], bbox_params=A.BboxParams(
            format='yolo',
            min_area=1,
            min_visibility=0.1,
            label_fields=['class_labels']
        ))
        
        # 颜色增强管道
        self.color_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.CLAHE(p=0.2),
            A.RandomGamma(p=0.2),
            A.ChannelShuffle(p=0.1),
        ])
        
        # 噪声和模糊增强管道
        self.noise_transform = A.Compose([
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            A.MotionBlur(blur_limit=7, p=0.2),
        ])
        
        # 天气效果增强
        self.weather_transform = A.Compose([
            A.RandomRain(p=0.1),
            A.RandomFog(p=0.1),
            A.RandomSunFlare(p=0.1),
            A.RandomShadow(p=0.1),
        ])
    
    def parse_yolo_annotation(self, annotation_path: str) -> Tuple[List, List]:
        """解析YOLO标注文件"""
        bboxes = []
        class_labels = []
        
        if not os.path.exists(annotation_path):
            return bboxes, class_labels
            
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            data = line.strip().split()
            if len(data) == 5:
                class_id = int(data[0])
                x_center, y_center, width, height = map(float, data[1:5])
                
                # 确保边界框在有效范围内
                if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
                
        return bboxes, class_labels
    
    def save_yolo_annotation(self, bboxes: List, class_labels: List, save_path: str):
        """保存YOLO格式标注"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                # 确保边界框在有效范围内
                x_center, y_center, width, height = bbox
                if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    f.write(line)
    
    def load_image_and_annotation(self, image_path: str, annotation_path: str) -> Tuple[np.ndarray, List, List]:
        """加载图像和标注"""
        # 读取图像
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取标注
        bboxes, class_labels = self.parse_yolo_annotation(annotation_path)
        
        return image, bboxes, class_labels
    
    def apply_base_transform(self, image: np.ndarray, bboxes: List, class_labels: List) -> Tuple[np.ndarray, List, List]:
        """应用基础变换（调整尺寸）"""
        if len(bboxes) == 0:
            transformed = self.base_transform(image=image)
            return transformed['image'], [], []
        else:
            transformed = self.base_transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            return transformed['image'], transformed['bboxes'], transformed['class_labels']
    
    def augment_single_image(self, image_path: str, annotation_path: str, 
                           use_geometric=True, use_color=True, use_noise=True, use_weather=False) -> Tuple[np.ndarray, List, List]:
        """对单张图像进行增强"""
        # 加载图像和标注
        image, bboxes, class_labels = self.load_image_and_annotation(image_path, annotation_path)
        
        # 应用基础变换
        image, bboxes, class_labels = self.apply_base_transform(image, bboxes, class_labels)
        
        # 应用几何变换
        if use_geometric and len(bboxes) > 0:
            try:
                transformed = self.geometric_transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                image, bboxes, class_labels = transformed['image'], transformed['bboxes'], transformed['class_labels']
            except Exception as e:
                print(f"几何变换失败: {e}")
        
        # 应用颜色变换
        if use_color:
            try:
                transformed = self.color_transform(image=image)
                image = transformed['image']
            except Exception as e:
                print(f"颜色变换失败: {e}")
        
        # 应用噪声变换
        if use_noise:
            try:
                transformed = self.noise_transform(image=image)
                image = transformed['image']
            except Exception as e:
                print(f"噪声变换失败: {e}")
        
        # 应用天气效果
        if use_weather:
            try:
                transformed = self.weather_transform(image=image)
                image = transformed['image']
            except Exception as e:
                print(f"天气效果失败: {e}")
        
        return image, bboxes, class_labels

class MosaicAugmentor:
    """Mosaic数据增强 - 4张图像拼接"""
    
    def __init__(self, image_size=640):
        self.image_size = image_size
    
    def create_mosaic(self, image_paths: List[str], annotation_paths: List[str]) -> Tuple[np.ndarray, List, List]:
        """创建4张图像的mosaic拼接"""
        if len(image_paths) != 4 or len(annotation_paths) != 4:
            raise ValueError("Mosaic增强需要恰好4张图像和4个标注文件")
        
        # 创建2x2的mosaic画布
        mosaic_image = np.zeros((self.image_size * 2, self.image_size * 2, 3), dtype=np.uint8)
        mosaic_bboxes = []
        mosaic_labels = []
        
        # 四个区域的坐标
        positions = [
            (0, 0),  # 左上
            (self.image_size, 0),  # 右上
            (0, self.image_size),  # 左下
            (self.image_size, self.image_size)  # 右下
        ]
        
        augmentor = YOLOAugmentor(self.image_size)
        
        for i, (img_path, ann_path) in enumerate(zip(image_paths, annotation_paths)):
            # 加载并调整图像大小
            image, bboxes, labels = augmentor.load_image_and_annotation(img_path, ann_path)
            image, bboxes, labels = augmentor.apply_base_transform(image, bboxes, labels)
            
            # 获取位置
            x_offset, y_offset = positions[i]
            
            # 将图像放置到mosaic中
            mosaic_image[y_offset:y_offset+self.image_size, 
                        x_offset:x_offset+self.image_size] = image
            
            # 转换边界框坐标
            for bbox, label in zip(bboxes, labels):
                x_center, y_center, width, height = bbox
                
                # 转换为绝对坐标（相对于原始图像）
                x_center_abs = x_center * self.image_size
                y_center_abs = y_center * self.image_size
                width_abs = width * self.image_size
                height_abs = height * self.image_size
                
                # 计算在mosaic图像中的新坐标
                new_x_center = (x_offset + x_center_abs) / (self.image_size * 2)
                new_y_center = (y_offset + y_center_abs) / (self.image_size * 2)
                new_width = width_abs / (self.image_size * 2)
                new_height = height_abs / (self.image_size * 2)
                
                # 确保边界框在有效范围内
                if (0 <= new_x_center <= 1 and 0 <= new_y_center <= 1 and 
                    0 <= new_width <= 1 and 0 <= new_height <= 1):
                    mosaic_bboxes.append([new_x_center, new_y_center, new_width, new_height])
                    mosaic_labels.append(label)
        
        return mosaic_image, mosaic_bboxes, mosaic_labels

class MixUpAugmentor:
    """MixUp数据增强 - 两张图像混合"""
    
    def __init__(self, image_size=640):
        self.image_size = image_size
    
    def create_mixup(self, image_path1: str, annotation_path1: str, 
                    image_path2: str, annotation_path2: str, 
                    alpha=0.5) -> Tuple[np.ndarray, List, List]:
        """创建MixUp混合图像"""
        augmentor = YOLOAugmentor(self.image_size)
        
        # 加载第一张图像
        image1, bboxes1, labels1 = augmentor.load_image_and_annotation(image_path1, annotation_path1)
        image1, bboxes1, labels1 = augmentor.apply_base_transform(image1, bboxes1, labels1)
        
        # 加载第二张图像
        image2, bboxes2, labels2 = augmentor.load_image_and_annotation(image_path2, annotation_path2)
        image2, bboxes2, labels2 = augmentor.apply_base_transform(image2, bboxes2, labels2)
        
        # 随机生成混合系数
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5
        
        # 混合图像
        mixed_image = (lam * image1 + (1 - lam) * image2).astype(np.uint8)
        
        # 合并边界框和标签
        mixed_bboxes = bboxes1 + bboxes2
        mixed_labels = labels1 + labels2
        
        return mixed_image, mixed_bboxes, mixed_labels

class YOLODatasetAugmentor:
    """YOLO数据集增强器"""
    
    def __init__(self, image_size=640, augmentations_per_image=5, 
                 use_mosaic=True, use_mixup=True, mosaic_prob=0.3, mixup_prob=0.3):
        self.image_size = image_size
        self.augmentations_per_image = augmentations_per_image
        self.use_mosaic = use_mosaic
        self.use_mixup = use_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        
        self.single_augmentor = YOLOAugmentor(image_size)
        self.mosaic_augmentor = MosaicAugmentor(image_size)
        self.mixup_augmentor = MixUpAugmentor(image_size)
    
    def augment_dataset(self, dataset_dir: str, output_dir: str, 
                       copy_original=True, image_extensions=None):
        """
        增强整个YOLO数据集
        
        Args:
            dataset_dir: 原始数据集目录
            output_dir: 增强后数据保存目录
            copy_original: 是否复制原始数据
            image_extensions: 图像文件扩展名列表
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        dataset_path = Path(dataset_dir)
        output_path = Path(output_dir)
        
        # 创建输出目录
        (output_path / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / 'labels').mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        image_files = self._find_image_files(dataset_path, image_extensions)
        annotation_files = [self._find_annotation_file(img_path, dataset_path) for img_path in image_files]
        
        # 过滤掉没有标注文件的图像
        valid_pairs = [(img, ann) for img, ann in zip(image_files, annotation_files) if ann.exists()]
        
        print(f"找到 {len(valid_pairs)} 张有效图像进行增强")
        
        if copy_original:
            # 复制原始数据
            self._copy_original_data(dataset_path, output_path, image_extensions)
        
        # 对每张图像进行增强
        total_augmented = 0
        
        for i, (image_file, annotation_file) in enumerate(valid_pairs):
            print(f"处理图像 {i+1}/{len(valid_pairs)}: {image_file.name}")
            
            for aug_idx in range(self.augmentations_per_image):
                try:
                    # 随机选择增强方式
                    augmentation_type = self._select_augmentation_type()
                    
                    if augmentation_type == "mosaic" and len(valid_pairs) >= 4:
                        # Mosaic增强
                        other_indices = random.sample([j for j in range(len(valid_pairs)) if j != i], 3)
                        mosaic_images = [image_file] + [valid_pairs[idx][0] for idx in other_indices]
                        mosaic_annotations = [annotation_file] + [valid_pairs[idx][1] for idx in other_indices]
                        
                        augmented_image, augmented_bboxes, augmented_labels = \
                            self.mosaic_augmentor.create_mosaic(mosaic_images, mosaic_annotations)
                        
                        aug_suffix = f"_mosaic_{aug_idx}"
                        
                    elif augmentation_type == "mixup" and len(valid_pairs) >= 2:
                        # MixUp增强
                        other_index = random.choice([j for j in range(len(valid_pairs)) if j != i])
                        mixup_image = valid_pairs[other_index][0]
                        mixup_annotation = valid_pairs[other_index][1]
                        
                        augmented_image, augmented_bboxes, augmented_labels = \
                            self.mixup_augmentor.create_mixup(
                                str(image_file), str(annotation_file),
                                str(mixup_image), str(mixup_annotation)
                            )
                        
                        aug_suffix = f"_mixup_{aug_idx}"
                    
                    else:
                        # 单图增强
                        augmented_image, augmented_bboxes, augmented_labels = \
                            self.single_augmentor.augment_single_image(
                                str(image_file), str(annotation_file)
                            )
                        
                        aug_suffix = f"_aug_{aug_idx}"
                    
                    # 保存增强后的图像和标注
                    aug_image_name = f"{image_file.stem}{aug_suffix}{image_file.suffix}"
                    aug_label_name = f"{image_file.stem}{aug_suffix}.txt"
                    
                    # 保存图像
                    aug_image_path = output_path / 'images' / aug_image_name
                    cv2.imwrite(str(aug_image_path), 
                               cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
                    
                    # 保存标注
                    aug_label_path = output_path / 'labels' / aug_label_name
                    self.single_augmentor.save_yolo_annotation(
                        augmented_bboxes, augmented_labels, str(aug_label_path)
                    )
                    
                    total_augmented += 1
                    
                except Exception as e:
                    print(f"增强 {image_file} 时出错: {e}")
                    continue
        
        print(f"增强完成！共生成 {total_augmented} 张增强图像")
    
    def _select_augmentation_type(self) -> str:
        """随机选择增强类型"""
        rand_val = random.random()
        
        if self.use_mosaic and rand_val < self.mosaic_prob:
            return "mosaic"
        elif self.use_mixup and rand_val < self.mosaic_prob + self.mixup_prob:
            return "mixup"
        else:
            return "single"
    
    def _find_image_files(self, dataset_path: Path, image_extensions: List[str]) -> List[Path]:
        """查找所有图像文件"""
        image_files = []
        
        # 在images目录中查找
        images_dir = dataset_path / 'images'
        if images_dir.exists():
            for ext in image_extensions:
                image_files.extend(images_dir.glob(f'*{ext}'))
                image_files.extend(images_dir.glob(f'*{ext.upper()}'))
        
        # 在根目录中查找
        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f'*{ext}'))
            image_files.extend(dataset_path.glob(f'*{ext.upper()}'))
        
        return list(set(image_files))  # 去重
    
    def _find_annotation_file(self, image_file: Path, dataset_path: Path) -> Path:
        """查找对应的标注文件"""
        # 尝试在labels目录中查找
        labels_dir = dataset_path / 'labels'
        if labels_dir.exists():
            label_path = labels_dir / f"{image_file.stem}.txt"
            if label_path.exists():
                return label_path
        
        # 尝试在同一目录中查找
        label_path = image_file.parent / f"{image_file.stem}.txt"
        if label_path.exists():
            return label_path
        
        # 返回不存在的路径（会被后续过滤掉）
        return dataset_path / 'labels' / f"{image_file.stem}.txt"
    
    def _copy_original_data(self, source_path: Path, dest_path: Path, image_extensions: List[str]):
        """复制原始数据到输出目录"""
        # 复制图像
        images_source = source_path / 'images'
        images_dest = dest_path / 'images'
        
        if images_source.exists():
            shutil.copytree(images_source, images_dest, dirs_exist_ok=True)
        else:
            for ext in image_extensions:
                for img_file in source_path.glob(f'*{ext}'):
                    shutil.copy2(img_file, images_dest)
                for img_file in source_path.glob(f'*{ext.upper()}'):
                    shutil.copy2(img_file, images_dest)
        
        # 复制标注
        labels_source = source_path / 'labels'
        labels_dest = dest_path / 'labels'
        
        if labels_source.exists():
            shutil.copytree(labels_source, labels_dest, dirs_exist_ok=True)
        else:
            for txt_file in source_path.glob('*.txt'):
                shutil.copy2(txt_file, labels_dest)

# 使用示例
def main():
    # 创建增强器
    augmentor = YOLODatasetAugmentor(
        image_size=640,
        augmentations_per_image=5,
        use_mosaic=True,
        use_mixup=True,
        mosaic_prob=0.2,
        mixup_prob=0.2
    )
    
    # 增强数据集
    augmentor.augment_dataset(
        dataset_dir='path/to/your/dataset',
        output_dir='path/to/augmented/dataset',
        copy_original=True
    )

# 单张图像测试
def test_single_augmentation():
    augmentor = YOLOAugmentor(image_size=640)
    
    image_path = 'test_image.jpg'
    annotation_path = 'test_annotation.txt'
    
    augmented_image, bboxes, labels = augmentor.augment_single_image(
        image_path, annotation_path
    )
    
    # 显示结果
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.imshow(original_image)
    plt.title('Original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(augmented_image)
    plt.title('Augmented')
    plt.show()

# Mosaic增强测试
def test_mosaic_augmentation():
    mosaic_augmentor = MosaicAugmentor(image_size=640)
    
    # 需要4张图像和对应的标注
    image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg']
    annotation_paths = ['annotation1.txt', 'annotation2.txt', 'annotation3.txt', 'annotation4.txt']
    
    mosaic_image, bboxes, labels = mosaic_augmentor.create_mosaic(image_paths, annotation_paths)
    
    # 显示结果
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(mosaic_image)
    plt.title('Mosaic Augmentation')
    plt.show()

if __name__ == "__main__":
    main()