import os
import cv2
import numpy as np
import albumentations as A
import random
import shutil
import time
from typing import List, Tuple
from pathlib import Path


class FixedYOLOAugmentor:
    def __init__(self, image_dir: str, label_dir: str, output_dir: str, 
                 min_bbox_area: float = 0.0001,  # 最小边界框面积（相对图像）
                 class_ids: List[int] = None):    # 已知有效类别ID（None则不校验）
        self.image_dir = Path(image_dir).resolve()
        self.label_dir = Path(label_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.min_bbox_area = min_bbox_area  # 过滤过小边界框（相对面积）
        self.class_ids = set(class_ids) if class_ids else None  # 有效类别ID集合
        
        # 创建输出目录
        self.output_img_dir = self.output_dir / 'images'
        self.output_label_dir = self.output_dir / 'labels'
        self.output_img_dir.mkdir(parents=True, exist_ok=True)
        self.output_label_dir.mkdir(parents=True, exist_ok=True)
        
        # 记录处理过的文件名，避免重复
        self.processed_names = set()

    def _get_matching_label(self, image_path: Path) -> Path:
        """获取与图像严格匹配的标注文件（处理多种扩展名和大小写）"""
        # 尝试常见标注文件格式：同文件名+.txt（忽略图像扩展名）
        label_stem = image_path.stem
        possible_labels = [
            self.label_dir / f"{label_stem}.txt",
            self.label_dir / f"{label_stem.lower()}.txt",
            self.label_dir / f"{label_stem.upper()}.txt"
        ]
        for lbl in possible_labels:
            if lbl.exists():
                return lbl
        return None

    def read_yolo_annotations(self, label_path: Path, img_width: int, img_height: int) -> List[Tuple]:
        """读取YOLO格式标注并进行严格验证"""
        annotations = []
        if not label_path.exists():
            return annotations
            
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    print(f"警告: {label_path} 第{line_num}行格式错误（需5个字段），跳过")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center_rel = float(parts[1])
                    y_center_rel = float(parts[2])
                    width_rel = float(parts[3])
                    height_rel = float(parts[4])
                    
                    # 1. 验证坐标范围（严格0-1）
                    if not (0 <= x_center_rel <= 1 and 0 <= y_center_rel <= 1):
                        print(f"警告: {label_path} 第{line_num}行中心坐标超出范围，跳过")
                        continue
                    if not (0 < width_rel <= 1 and 0 < height_rel <= 1):
                        print(f"警告: {label_path} 第{line_num}行宽高超出范围，跳过")
                        continue
                    
                    # 2. 验证边界框面积（避免过小框）
                    bbox_area = width_rel * height_rel
                    if bbox_area < self.min_bbox_area:
                        print(f"警告: {label_path} 第{line_num}行边界框过小（面积{bbox_area:.6f}），跳过")
                        continue
                    
                    # 3. 验证类别ID（如果已知有效类别）
                    if self.class_ids is not None and class_id not in self.class_ids:
                        print(f"警告: {label_path} 第{line_num}行类别ID {class_id} 无效，跳过")
                        continue
                    
                    annotations.append((class_id, x_center_rel, y_center_rel, width_rel, height_rel))
                    
                except ValueError as e:
                    print(f"警告: {label_path} 第{line_num}行数值解析错误: {e}，跳过")
                    continue
                    
        except Exception as e:
            print(f"错误: 读取标注 {label_path} 失败: {e}")
            
        return annotations

    def convert_to_yolo_format(self, bboxes: List[Tuple]) -> List[str]:
        """转换为YOLO格式并进行最终校验"""
        yolo_lines = []
        for bbox in bboxes:
            class_id, x_center, y_center, width, height = bbox
            
            # 强制裁剪到有效范围（避免增强后微小越界）
            x_center = max(0.001, min(0.999, x_center))
            y_center = max(0.001, min(0.999, y_center))
            width = max(0.001, min(0.999 - x_center*2, width))  # 确保不超出边界
            height = max(0.001, min(0.999 - y_center*2, height))
            
            # 再次验证面积
            if width * height < self.min_bbox_area:
                print(f"过滤增强后过小边界框（面积{width*height:.6f}）")
                continue
            
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
        return yolo_lines

    def get_augmentation(self, aug_type: str):
        """获取增强变换（添加边界框裁剪保护）"""
        bbox_params = A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.5,  # 边界框可见度低于50%则丢弃
            min_area=self.min_bbox_area  # 增强后最小面积（相对原图像）
        )
        
        if aug_type == "light":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.2),
                ], p=0.1)
            ], bbox_params=bbox_params)
        
        elif aug_type == "medium":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=5, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            ], bbox_params=bbox_params)
        
        else:  # heavy
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
                A.OneOf([
                    A.Blur(blur_limit=3, p=0.2),
                    A.GaussianBlur(blur_limit=3, p=0.2),
                ], p=0.2),
                A.CLAHE(clip_limit=2.0, p=0.2),
            ], bbox_params=bbox_params)

    def _check_duplicate_name(self, output_name: str) -> str:
        """确保输出文件名唯一（避免重复）"""
        original_name = output_name
        suffix = 1
        while output_name in self.processed_names:
            output_name = f"{original_name}_{suffix}"
            suffix += 1
        self.processed_names.add(output_name)
        return output_name

    def augment_image(self, image_path: Path, label_path: Path, aug_type: str, base_name: str) -> bool:
        """增强单张图像（强化错误处理和有效性验证）"""
        try:
            # 1. 验证图像有效性
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"错误: 无法读取图像 {image_path}，跳过")
                return False
            if image.shape[0] < 32 or image.shape[1] < 32:  # 过滤过小图像
                print(f"错误: 图像 {image_path} 尺寸过小（<32x32），跳过")
                return False
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            
            # 2. 读取并验证标注
            annotations = self.read_yolo_annotations(label_path, width, height)
            if not annotations:
                print(f"警告: 图像 {image_path} 无有效标注，跳过增强")
                return False
            
            # 3. 准备增强数据
            bboxes = []
            class_labels = []
            for ann in annotations:
                class_id, x_center_rel, y_center_rel, width_rel, height_rel = ann
                bboxes.append([x_center_rel, y_center_rel, width_rel, height_rel])
                class_labels.append(class_id)
            
            # 4. 应用增强
            transform = self.get_augmentation(aug_type)
            try:
                transformed = transform(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
            except Exception as e:
                print(f"增强变换失败 {image_path}: {e}，跳过")
                return False
            
            # 5. 验证增强后边界框
            transformed_bboxes = []
            for bbox, class_id in zip(transformed['bboxes'], transformed['class_labels']):
                x_center, y_center, bbox_width, bbox_height = bbox
                # 严格验证范围
                if not (0.001 <= x_center <= 0.999 and 0.001 <= y_center <= 0.999):
                    continue
                if not (0.001 <= bbox_width <= 0.999 and 0.001 <= bbox_height <= 0.999):
                    continue
                # 验证面积
                if bbox_width * bbox_height < self.min_bbox_area:
                    continue
                transformed_bboxes.append((class_id, x_center, y_center, bbox_width, bbox_height))
            
            if not transformed_bboxes:
                print(f"增强后无有效边界框 {image_path}，跳过")
                return False
            
            # 6. 生成唯一文件名并保存
            output_name = self._check_duplicate_name(f"{base_name}_{aug_type}")
            yolo_lines = self.convert_to_yolo_format(transformed_bboxes)
            
            # 保存图像
            output_image_path = self.output_img_dir / f"{output_name}.jpg"
            if not cv2.imwrite(str(output_image_path), cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)):
                print(f"错误: 无法保存增强图像 {output_image_path}，跳过")
                return False
            
            # 保存标注
            output_label_path = self.output_label_dir / f"{output_name}.txt"
            with open(output_label_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
            
            return True
            
        except Exception as e:
            print(f"增强图像 {image_path} 出错: {e}，跳过")
            return False

    def create_mosaic(self, image_paths: List[Path], label_paths: List[Path], output_name: str) -> bool:
        """创建马赛克增强（统一图像尺寸，强化校验）"""
        try:
            if len(image_paths) != 4:
                return False
            
            images = []
            all_annotations = []
            target_size = None  # 统一马赛克子图尺寸
            
            # 1. 读取并预处理4张图像（统一尺寸）
            for img_path, label_path in zip(image_paths, label_paths):
                # 读取图像
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"马赛克错误: 无法读取 {img_path}，跳过")
                    return False
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image_rgb.shape[:2]
                
                # 统一尺寸（以第一张图像为基准）
                if target_size is None:
                    target_size = (w, h)  # (width, height)
                else:
                    if (w, h) != target_size:
                        # 缩放至目标尺寸（避免马赛克拼接变形）
                        image_rgb = cv2.resize(image_rgb, target_size)
                        h, w = target_size[1], target_size[0]  # resize后h=target_height, w=target_width
                
                images.append(image_rgb)
                
                # 读取标注
                annotations = self.read_yolo_annotations(label_path, w, h)
                if not annotations:
                    print(f"马赛克警告: {img_path} 无有效标注，跳过此组合")
                    return False
                all_annotations.append(annotations)
            
            # 2. 创建马赛克画布（2x2拼接）
            mosaic_width = 2 * target_size[0]
            mosaic_height = 2 * target_size[1]
            mosaic_image = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
            
            # 3. 拼接图像并转换标注
            positions = [
                (0, 0),  # 左上
                (target_size[0], 0),  # 右上
                (0, target_size[1]),  # 左下  
                (target_size[0], target_size[1])  # 右下
            ]
            
            mosaic_annotations = []
            for img, annotations, (x, y) in zip(images, all_annotations, positions):
                h, w = img.shape[:2]
                mosaic_image[y:y+h, x:x+w] = img  # 拼接图像
                
                # 转换标注坐标
                for ann in annotations:
                    class_id, x_center_rel, y_center_rel, width_rel, height_rel = ann
                    
                    # 计算马赛克中的绝对坐标
                    x_center_abs = x_center_rel * w + x
                    y_center_abs = y_center_rel * h + y
                    width_abs = width_rel * w
                    height_abs = height_rel * h
                    
                    # 转换为相对马赛克的坐标
                    x_center_new = x_center_abs / mosaic_width
                    y_center_new = y_center_abs / mosaic_height
                    width_new = width_abs / mosaic_width
                    height_new = height_abs / mosaic_height
                    
                    # 验证有效性
                    if (0.001 <= x_center_new <= 0.999 and 
                        0.001 <= y_center_new <= 0.999 and 
                        0.001 <= width_new <= 0.999 and 
                        0.001 <= height_new <= 0.999 and 
                        width_new * height_new >= self.min_bbox_area):
                        mosaic_annotations.append((class_id, x_center_new, y_center_new, width_new, height_new))
            
            if not mosaic_annotations:
                print(f"马赛克无有效标注 {output_name}，跳过")
                return False
            
            # 4. 保存马赛克结果
            output_name = self._check_duplicate_name(output_name)
            yolo_lines = self.convert_to_yolo_format(mosaic_annotations)
            
            output_image_path = self.output_img_dir / f"{output_name}.jpg"
            if not cv2.imwrite(str(output_image_path), cv2.cvtColor(mosaic_image, cv2.COLOR_RGB2BGR)):
                print(f"错误: 无法保存马赛克图像 {output_image_path}，跳过")
                return False
            
            output_label_path = self.output_label_dir / f"{output_name}.txt"
            with open(output_label_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
            
            return True
            
        except Exception as e:
            print(f"马赛克创建错误: {e}，跳过")
            return False

    def augment_to_1000(self):
        """增强到1000张（优化流程和校验）"""
        # 1. 获取并校验图像-标注对（严格匹配）
        valid_pairs = []  # 存储(图像路径, 标注路径)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for img_path in self.image_dir.glob('*.*'):
            if img_path.suffix.lower() not in image_extensions:
                continue
            # 严格匹配标注文件
            label_path = self._get_matching_label(img_path)
            if label_path is None:
                print(f"警告: 图像 {img_path.name} 无匹配标注文件，跳过")
                continue
            valid_pairs.append((img_path, label_path))
        
        original_count = len(valid_pairs)
        print(f"有效图像-标注对数量: {original_count}")
        
        if original_count == 0:
            print("❌ 没有找到有效图像-标注对")
            return
        
        # 2. 复制原始数据（带校验）
        print("📁 复制原始图片和标注...")
        copied_count = 0
        for img_path, label_path in valid_pairs:
            # 生成唯一文件名
            base_name = f"original_{img_path.stem}"
            output_name = self._check_duplicate_name(base_name)
            
            # 复制图像
            img_dst = self.output_img_dir / f"{output_name}{img_path.suffix}"
            if not img_dst.exists():
                try:
                    shutil.copy2(img_path, img_dst)
                except Exception as e:
                    print(f"复制图像 {img_path} 失败: {e}，跳过")
                    continue
            
            # 复制标注
            label_dst = self.output_label_dir / f"{output_name}.txt"
            if not label_dst.exists():
                try:
                    shutil.copy2(label_path, label_dst)
                except Exception as e:
                    print(f"复制标注 {label_path} 失败: {e}，删除对应图像")
                    if img_dst.exists():
                        os.remove(img_dst)
                    continue
            
            copied_count += 1
        
        print(f"✅ 复制原始数据: {copied_count} 对")
        if copied_count == 0:
            print("❌ 无法复制原始数据，终止增强")
            return
        
        # 3. 基础增强（每张生成多个版本）
        print("🔄 开始基础增强...")
        base_count = 0
        augmentation_types = ['light', 'medium', 'heavy']
        versions_per_type = 2  # 每种增强类型生成的版本数
        
        for i, (img_path, label_path) in enumerate(valid_pairs):
            base_name = img_path.stem
            # 每种增强类型生成多个版本
            for aug_type in augmentation_types:
                for ver in range(versions_per_type):
                    success = self.augment_image(img_path, label_path, aug_type, f"{base_name}_v{ver+1}")
                    if success:
                        base_count += 1
            
            # 进度提示
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i+1}/{original_count} 张，生成基础增强 {base_count} 张")
        
        print(f"✅ 基础增强完成: {base_count} 张")
        
        # 4. 马赛克增强（选择有效图像组合）
        print("🧩 开始马赛克增强...")
        mosaic_count = 0
        # 计算需要的马赛克数量（避免过度生成）
        current_total = copied_count + base_count
        remaining = 1000 - current_total
        target_mosaic = min(200, max(0, remaining // 2))  # 马赛克占比不超过200
        
        for i in range(target_mosaic):
            if len(valid_pairs) < 4:
                break
            # 随机选择4个不同的图像-标注对
            selected = random.sample(valid_pairs, 4)
            selected_imgs = [p[0] for p in selected]
            selected_labels = [p[1] for p in selected]
            
            success = self.create_mosaic(selected_imgs, selected_labels, f"mosaic_{i+1:04d}")
            if success:
                mosaic_count += 1
            
            if (i + 1) % 20 == 0:
                print(f"  已生成马赛克 {i+1}/{target_mosaic} 张")
        
        print(f"✅ 马赛克增强完成: {mosaic_count} 张")
        
        # 5. 额外增强（补充至1000张）
        current_total += mosaic_count
        remaining = 1000 - current_total
        
        if remaining > 0:
            print(f"➕ 需要额外生成 {remaining} 张...")
            extra_count = 0
            augmentation_types = ['light', 'medium', 'heavy']
            
            for i in range(remaining):
                # 随机选择图像
                img_path, label_path = random.choice(valid_pairs)
                base_name = img_path.stem
                aug_type = random.choice(augmentation_types)
                
                success = self.augment_image(img_path, label_path, aug_type, f"extra_{i+1:04d}_{base_name}")
                if success:
                    extra_count += 1
                    # 每生成100张检查是否已达标
                    if (copied_count + base_count + mosaic_count + extra_count) >= 1000:
                        break
                
                if (i + 1) % 50 == 0:
                    print(f"  已生成额外增强 {i+1}/{remaining} 张")
            
            print(f"✅ 额外增强完成: {extra_count} 张")
        else:
            extra_count = 0
        
        # 6. 最终校验（确保图像和标注一一对应）
        self._final_verification()

    def _final_verification(self):
        """最终验证增强数据的完整性和有效性"""
        print("\n" + "="*60)
        print("🔍 增强数据最终验证")
        print("="*60)
        
        # 统计输出文件
        images = [f for f in self.output_img_dir.glob('*.*') 
                 if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        labels = [f for f in self.output_label_dir.glob('*.txt')]
        
        # 提取文件名（不含扩展名）
        img_names = {f.stem for f in images}
        label_names = {f.stem for f in labels}
        
        # 检查匹配性
        missing_labels = img_names - label_names  # 有图像无标注
        missing_images = label_names - img_names  # 有标注无图像
        
        print(f"总图像数: {len(images)}")
        print(f"总标注数: {len(labels)}")
        
        if not missing_labels and not missing_images:
            print("✅ 图像和标注完全匹配")
        else:
            if missing_labels:
                print(f"❌ 有 {len(missing_labels)} 张图像缺少标注，已自动删除")
                for name in missing_labels:
                    for ext in ['.jpg', '.jpeg', '.png']:
                        img_path = self.output_img_dir / f"{name}{ext}"
                        if img_path.exists():
                            os.remove(img_path)
            if missing_images:
                print(f"❌ 有 {len(missing_images)} 个标注缺少图像，已自动删除")
                for name in missing_images:
                    label_path = self.output_label_dir / f"{name}.txt"
                    if label_path.exists():
                        os.remove(label_path)
        
        # 重新统计
        final_images = len([f for f in self.output_img_dir.glob('*.*') 
                          if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        final_labels = len(list(self.output_label_dir.glob('*.txt')))
        
        print("\n" + "="*60)
        print("🎉 增强完成统计")
        print("="*60)
        print(f"原始数据: {len([f for f in images if f.stem.startswith('original_')])} 张")
        print(f"基础增强: {len([f for f in images if any(t in f.stem for t in ['light', 'medium', 'heavy']) and 'extra' not in f.stem])} 张")
        print(f"马赛克增强: {len([f for f in images if 'mosaic' in f.stem])} 张")
        print(f"额外增强: {len([f for f in images if 'extra' in f.stem])} 张")
        print(f"最终总数: {final_images} 张 (目标: 1000 张)")
        
        if final_images >= 1000:
            print("✅ 成功达到目标数量！")
        else:
            print(f"⚠️  未达到目标，当前 {final_images}/1000 张")
        
        print("="*60)


def main():
    # 配置路径（可根据实际情况修改）
    IMAGE_DIR = "D:\Galaxy\其他\桌面\yolo_data\images"
    LABEL_DIR = "D:\Galaxy\其他\桌面\yolo_data\labels"
    OUTPUT_DIR = "augmented_1000_fixed"
    
    # 可选配置：已知有效类别ID（例如[0,1,2]），None则不校验
    VALID_CLASS_IDS = 0  # 示例：[0, 1, 2]
    # 最小边界框相对面积（例如0.0001表示图像的0.01%）
    MIN_BBOX_AREA = 0.0001
    
    print("=" * 60)
    print("YOLO数据增强工具（增强版）")
    print("=" * 60)
    
    # 检查输入目录
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ 错误: 图像目录不存在 {IMAGE_DIR}")
        return
    if not os.path.exists(LABEL_DIR):
        print(f"❌ 错误: 标注目录不存在 {LABEL_DIR}")
        return
    
    # 创建增强器
    augmentor = FixedYOLOAugmentor(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        output_dir=OUTPUT_DIR,
        min_bbox_area=MIN_BBOX_AREA,
        class_ids=VALID_CLASS_IDS
    )
    
    # 开始增强
    start_time = time.time()
    augmentor.augment_to_1000()
    end_time = time.time()
    
    # 显示耗时
    duration = end_time - start_time
    print(f"⏱️  总耗时: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
    print(f"💾 输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()