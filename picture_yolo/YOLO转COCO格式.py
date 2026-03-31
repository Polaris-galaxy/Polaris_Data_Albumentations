import os
import json
import argparse
import shutil
from pathlib import Path
from PIL import Image
import yaml
from datetime import datetime


class YOLOToCOCOConverter:
    def __init__(self, yolo_path, output_dir, copy_images=False):
        self.yolo_path = Path(yolo_path).resolve()  # 绝对路径，避免歧义
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.copy_images = copy_images  # 是否复制图像到输出目录
        
        # 初始化COCO完整格式（包含所有必填字段）
        self.coco_format = {
            "info": {
                "description": "Converted from YOLO format",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "YOLO to COCO Converter",
                "date_created": datetime.now().strftime("%Y-%m-%d")
            },
            "licenses": [
                {
                    "url": "",
                    "id": 1,
                    "name": "Unknown License"
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        self.annotation_id = 1
        self.image_id = 1
        self.class_names = []  # 类别名称列表
        self.all_class_ids = set()  # 收集所有出现过的类别ID（用于校验）

    def load_class_names(self):
        """加载类别名称（优先从data.yaml，否则自动推断，兜底虚拟类别）"""
        data_yaml_path = self.yolo_path / 'data.yaml'
        if data_yaml_path.exists():
            # 尝试多种编码读取data.yaml（解决中文/特殊字符编码问题）
            data = self._load_yaml_with_encoding(data_yaml_path)
            if data is None:
                print("错误：data.yaml存在但无法解析，将扫描标签文件推断类别")
                # 不返回False，继续执行标签扫描
            else:
                # 校验data.yaml关键字段
                if 'names' not in data:
                    print("警告：data.yaml中未找到'names'字段，将扫描标签文件推断类别")
                else:
                    self.class_names = data['names']
                    # 校验nc与names长度一致性
                    if 'nc' in data and data['nc'] != len(self.class_names):
                        print(f"警告：data.yaml中'nc'（{data['nc']}）与'names'长度（{len(self.class_names)}）不一致，以'names'为准")
                    # 初始化有效类别ID（0到len(names)-1）
                    self.all_class_ids = set(range(len(self.class_names)))
                    print(f"从data.yaml加载类别: {self.class_names}")
                    print(f"有效类别ID范围: 0 ~ {len(self.class_names)-1}")
                    return True
        
        # 无data.yaml或读取失败时，扫描标签文件推断类别
        print("未找到有效data.yaml，开始扫描所有标签文件推断类别...")
        splits = ['train', 'val', 'test']
        for split in splits:
            labels_dir = self.yolo_path / 'labels' / split
            if labels_dir.exists():
                self._collect_class_ids(labels_dir)
        
        # 兜底逻辑：如果未收集到任何类别ID（所有txt都空或无有效标注）
        if not self.all_class_ids:
            print("警告：未找到任何有效标注数据，添加虚拟类别'background'（ID=0）")
            self.all_class_ids = {0}
            self.class_names = ['background']
            return True
        
        # 生成类别名称（按ID排序）
        sorted_ids = sorted(self.all_class_ids)
        self.class_names = [f"class_{i}" for i in sorted_ids] if not self.class_names else self.class_names
        print(f"推断类别（ID:名称）: {[(i, self.class_names[i]) for i in sorted_ids]}")
        return True

    def _load_yaml_with_encoding(self, yaml_path):
        """尝试多种编码读取YAML文件，避免编码错误"""
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'ansi']
        for encoding in encodings:
            try:
                with open(yaml_path, 'r', encoding=encoding) as f:
                    return yaml.safe_load(f)
            except (UnicodeDecodeError, yaml.YAMLError) as e:
                continue
        print(f"错误：无法解析YAML文件 {yaml_path}（尝试编码：{encodings}）")
        return None

    def _collect_class_ids(self, labels_dir):
        """扫描标签文件收集所有出现的类别ID（支持多编码，忽略空文件）"""
        for label_file in labels_dir.glob('*.txt'):
            try:
                # 尝试多种编码读取标签文件
                content = self._read_file_with_encoding(label_file)
                if content is None:
                    continue  # 读取失败则跳过
                
                # 过滤空行，检查是否为真正的空文件
                non_empty_lines = [line.strip() for line in content if line.strip()]
                if not non_empty_lines:
                    print(f"提示：标签文件 {label_file.name} 为空，跳过（对应图像无标注）")
                    continue
                
                # 解析有效行的class_id
                for line in non_empty_lines:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        try:
                            class_id = int(float(parts[0]))  # 兼容浮点数ID（如0.0→0）
                            self.all_class_ids.add(class_id)
                        except ValueError:
                            continue  # 非数字ID则跳过
            except Exception as e:
                print(f"警告：处理标签文件 {label_file} 时出错: {str(e)}")

    def _read_file_with_encoding(self, file_path):
        """尝试多种编码读取文本文件"""
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'ansi']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.readlines()
            except UnicodeDecodeError:
                continue
        print(f"警告：无法读取文件 {file_path}（尝试编码：{encodings}）")
        return None

    def get_image_size(self, image_path):
        """获取图像尺寸（增强错误处理）"""
        try:
            with Image.open(image_path) as img:
                return img.size  # (width, height)
        except Exception as e:
            print(f"错误：无法读取图像 {image_path} 的尺寸: {str(e)}")
            return None

    def convert_bbox_yolo_to_coco(self, bbox, img_width, img_height):
        """YOLO边界框（相对坐标）转COCO边界框（绝对坐标）"""
        x_center, y_center, width, height = bbox
        
        # 转换为绝对坐标
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height
        
        # 计算左上角坐标（确保不超出图像边界）
        x_min = max(0.0, x_center_abs - width_abs / 2)
        y_min = max(0.0, y_center_abs - height_abs / 2)
        width_abs = min(width_abs, img_width - x_min)  # 避免超出右边界
        height_abs = min(height_abs, img_height - y_min)  # 避免超出下边界
        
        return [x_min, y_min, width_abs, height_abs]

    def _copy_image_if_needed(self, image_path, split_name):
        """复制图像到输出目录（如果启用）"""
        if not self.copy_images:
            return str(image_path)  # 返回原路径
        
        # 输出图像路径：output_dir/images/split_name/xxx.jpg
        output_img_dir = self.output_dir / 'images' / split_name
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_img_path = output_img_dir / image_path.name
        
        if not output_img_path.exists():
            try:
                shutil.copy2(image_path, output_img_path)
            except Exception as e:
                print(f"警告：复制图像 {image_path} 失败: {str(e)}，将使用原路径")
                return str(image_path)
        return str(output_img_path.relative_to(self.output_dir))  # 相对输出目录的路径

    def process_split(self, split_name):
        """处理单个分割（train/val/test）"""
        print(f"\n===== 处理 {split_name} 分割 =====")
        images_dir = self.yolo_path / 'images' / split_name
        labels_dir = self.yolo_path / 'labels' / split_name
        
        # 检查图像目录是否存在
        if not images_dir.exists():
            print(f"跳过 {split_name}：图像目录不存在 {images_dir}")
            return
        
        # 筛选有效图像文件
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in images_dir.glob('*.*') 
                      if f.suffix.lower() in valid_extensions]
        
        if not image_files:
            print(f"跳过 {split_name}：未找到有效图像文件（支持格式：{valid_extensions}）")
            return
        
        print(f"找到 {len(image_files)} 张图像，开始处理...")
        processed = 0
        errors = 0
        skipped_annotations = 0  # 统计被跳过的标注数
        
        for img_path in image_files:
            # 获取图像尺寸
            img_size = self.get_image_size(img_path)
            if not img_size:
                errors += 1
                continue
            img_w, img_h = img_size
            
            # 复制图像（如果需要）并获取存储路径
            img_rel_path = self._copy_image_if_needed(img_path, split_name)
            
            # 添加图像信息到COCO（即使无标注也保留图像）
            self.coco_format["images"].append({
                "id": self.image_id,
                "file_name": img_rel_path,
                "width": img_w,
                "height": img_h,
                "license": 1,
                "date_captured": ""
            })
            
            # 处理对应标签文件
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                print(f"警告：{img_path.name} 无对应标签文件，仅保留图像信息")
                self.image_id += 1
                processed += 1
                continue
            
            # 读取标签内容（支持多编码）
            lines = self._read_file_with_encoding(label_path)
            if lines is None:
                print(f"警告：{img_path.name} 标签文件读取失败，仅保留图像信息")
                self.image_id += 1
                errors += 1
                continue
            
            # 解析标签行（忽略空行）
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            if not non_empty_lines:
                print(f"提示：{img_path.name} 标签文件为空，仅保留图像信息")
                self.image_id += 1
                processed += 1
                continue
            
            # 解析有效标注行
            for line in non_empty_lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    skipped_annotations += 1
                    print(f"警告：{label_path} 格式错误（需5个字段），跳过该行: {line.strip()}")
                    continue
                
                try:
                    class_id = int(float(parts[0]))
                    bbox_yolo = list(map(float, parts[1:5]))
                    
                    # 校验YOLO坐标是否在[0,1]范围内
                    if any(coord < 0 or coord > 1 for coord in bbox_yolo):
                        skipped_annotations += 1
                        print(f"警告：{img_path.name} 边界框坐标超出范围 {bbox_yolo}，跳过")
                        continue
                    
                    # 转换边界框格式
                    bbox_coco = self.convert_bbox_yolo_to_coco(bbox_yolo, img_w, img_h)
                    
                    # 计算面积（COCO要求）
                    area = bbox_coco[2] * bbox_coco[3]
                    
                    # 校验类别ID有效性（兼容虚拟类别）
                    if class_id not in self.all_class_ids:
                        skipped_annotations += 1
                        print(f"警告：{img_path.name} 类别ID {class_id} 不在已知类别中（有效ID：{sorted(self.all_class_ids)}），跳过")
                        continue
                    
                    # 添加标注
                    self.coco_format["annotations"].append({
                        "id": self.annotation_id,
                        "image_id": self.image_id,
                        "category_id": class_id + 1,  # COCO类别ID从1开始
                        "bbox": bbox_coco,
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": []  # 检测任务用空列表
                    })
                    self.annotation_id += 1
                    
                except (ValueError, IndexError) as e:
                    skipped_annotations += 1
                    print(f"警告：解析 {label_path} 失败: {str(e)}，跳过该行")
                    continue
            
            self.image_id += 1
            processed += 1
            if processed % 100 == 0:
                print(f"已处理 {processed}/{len(image_files)} 张图像（跳过标注：{skipped_annotations} 个）")
        
        print(f"{split_name} 处理完成：成功 {processed} 张，错误 {errors} 个，跳过标注 {skipped_annotations} 个")

    def create_categories(self):
        """创建COCO类别信息（确保ID连续）"""
        # 按类别ID排序，确保类别列表与ID对应
        if self.class_names:
            # 从data.yaml加载或兜底虚拟类别时，按names顺序生成
            sorted_class_ids = list(range(len(self.class_names)))
        else:
            # 自动推断时，按收集的ID排序
            sorted_class_ids = sorted(self.all_class_ids)
        
        for idx, class_id in enumerate(sorted_class_ids):
            self.coco_format["categories"].append({
                "id": idx + 1,  # COCO类别ID从1开始
                "name": self.class_names[idx] if self.class_names else f"class_{class_id}",
                "supercategory": "none"
            })

    def save_coco_json(self, split_name):
        """保存COCO格式JSON文件"""
        output_path = self.output_dir / f"instances_{split_name}.json"
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.coco_format, f, indent=2, ensure_ascii=False)
            
            print(f"\n已保存 {split_name} 数据到：{output_path}")
            print(f"  - 图像数量：{len(self.coco_format['images'])}")
            print(f"  - 标注数量：{len(self.coco_format['annotations'])}")
            print(f"  - 类别数量：{len(self.coco_format['categories'])}")
        except Exception as e:
            print(f"错误：保存 {split_name} 数据失败: {str(e)}")

    def convert(self):
        """执行完整转换流程"""
        # 加载类别名称（修改后不会返回False，确保流程继续）
        if not self.load_class_names():
            return False
        
        # 创建类别信息
        self.create_categories()
        
        # 处理所有分割（支持自定义分割，可在此处添加）
        splits = ['train', 'val', 'test']
        for split in splits:
            # 重置当前分割的图像和标注列表
            self.coco_format["images"] = []
            self.coco_format["annotations"] = []
            self.annotation_id = 1
            self.image_id = 1
            
            self.process_split(split)
            if self.coco_format["images"]:  # 只有当有图像时才保存
                self.save_coco_json(split)
            else:
                print(f"跳过保存 {split}：无有效图像")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="YOLO格式数据集转COCO格式（空文件兼容增强版）")
    parser.add_argument('--yolo_path', type=str, required=True, 
                       help="YOLO数据集根目录（需包含images/labels子目录）")
    parser.add_argument('--output_dir', type=str, required=True, 
                       help="COCO格式输出目录")
    parser.add_argument('--copy-images', action='store_true', 
                       help="是否将图像复制到输出目录（默认不复制）")
    
    args = parser.parse_args()
    
    # 初始化转换器
    try:
        converter = YOLOToCOCOConverter(
            yolo_path=args.yolo_path,
            output_dir=args.output_dir,
            copy_images=args.copy_images
        )
    except Exception as e:
        print(f"初始化失败：{str(e)}")
        return
    
    # 执行转换
    print("===== 开始YOLO到COCO格式转换 =====")
    success = converter.convert()
    
    if success:
        print("\n===== 转换成功！ =====")
        print(f"结果保存目录：{args.output_dir}")
        if args.copy_images:
            print("提示：图像已复制到输出目录的images子文件夹")
    else:
        print("\n===== 转换失败 =====")


if __name__ == "__main__":
    main()