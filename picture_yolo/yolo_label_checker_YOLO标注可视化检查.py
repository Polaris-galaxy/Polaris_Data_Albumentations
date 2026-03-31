import os
import cv2
import numpy as np
import random
from typing import List, Tuple
from pathlib import Path  # 新增：用Path处理路径更可靠
import traceback  # 新增：输出详细错误堆栈


class YOLOLabelChecker:
    def __init__(self, image_dir: str, label_dir: str):
        self.image_dir = Path(image_dir).resolve()  # 转为绝对路径，避免相对路径歧义
        self.label_dir = Path(label_dir).resolve()
        # 验证图像和标注目录是否存在（新增）
        if not self.image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在：{self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"标注目录不存在：{self.label_dir}")
        
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128)
        ]
    
    def read_yolo_annotations(self, label_path: Path) -> List[Tuple]:
        """读取YOLO标注文件（增强错误提示）"""
        annotations = []
        # 检查标注文件是否存在（新增详细路径提示）
        if not label_path.exists():
            print(f"❌ 标注文件不存在（路径：{label_path}）")
            return annotations
        
        # 尝试多种编码读取文件（解决编码问题）
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'ansi']  # 常见编码优先级
        content = None
        for encoding in encodings:
            try:
                with open(label_path, 'r', encoding=encoding) as f:
                    content = f.readlines()
                break  # 读取成功则退出编码尝试
            except UnicodeDecodeError:
                continue  # 编码错误则尝试下一种
            except Exception as e:
                print(f"❌ 读取文件时发生错误（路径：{label_path}，编码：{encoding}）：{str(e)}")
                continue
        
        if content is None:
            print(f"❌ 无法读取标注文件（路径：{label_path}），尝试过的编码：{encodings}")
            return annotations
        
        # 解析标注内容
        try:
            for line_num, line in enumerate(content, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行（不报错）
                
                parts = line.split()
                # 检查字段数量
                if len(parts) != 5:
                    print(f"❌ 格式错误（路径：{label_path}，行号：{line_num}）："
                          f"每行必须有5个字段，实际有{len(parts)}个，内容：{line}")
                    continue
                
                # 解析类别ID和坐标
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError as e:
                    print(f"❌ 数值格式错误（路径：{label_path}，行号：{line_num}）："
                          f"字段必须为数字，内容：{line}，错误：{str(e)}")
                    continue
                
                # 验证坐标范围
                coord_errors = []
                if not (0 <= x_center <= 1):
                    coord_errors.append(f"x_center={x_center}（需在0-1之间）")
                if not (0 <= y_center <= 1):
                    coord_errors.append(f"y_center={y_center}（需在0-1之间）")
                if not (0 < width <= 1):
                    coord_errors.append(f"width={width}（需在0-1之间且>0）")
                if not (0 < height <= 1):
                    coord_errors.append(f"height={height}（需在0-1之间且>0）")
                
                if coord_errors:
                    print(f"❌ 坐标范围错误（路径：{label_path}，行号：{line_num}）："
                          f"{', '.join(coord_errors)}，内容：{line}")
                    continue
                
                # 所有检查通过，添加到有效标注
                annotations.append((class_id, x_center, y_center, width, height))
                
        except Exception as e:
            print(f"❌ 解析标注文件时发生未知错误（路径：{label_path}）")
            print("详细错误堆栈：")
            print(traceback.format_exc())  # 输出完整错误堆栈，方便定位问题
            
        return annotations
    
    def visualize_single_image(self, image_file: str, save_dir: str = None):
        """可视化单张图像的标签（增强路径提示）"""
        # 构建图像和标注文件的完整路径
        image_path = self.image_dir / image_file  # 用Path拼接，避免路径错误
        if not image_path.exists():
            print(f"❌ 图像文件不存在（路径：{image_path}）")
            return False
        
        # 生成标注文件名（用splitext处理，支持多扩展名如.jpg/.jpeg/.png）
        label_filename = f"{os.path.splitext(image_file)[0]}.txt"
        label_path = self.label_dir / label_filename
        
        # 读取图像
        image = cv2.imread(str(image_path))  # Path转字符串
        if image is None:
            print(f"❌ 无法读取图像（可能格式错误或损坏，路径：{image_path}）")
            return False
        
        h, w = image.shape[:2]
        
        # 读取标注（传入Path对象）
        annotations = self.read_yolo_annotations(label_path)
        print(f"📌 图像：{image_file}（路径：{image_path}），找到 {len(annotations)} 个有效标注")
        
        # 绘制边界框（保持原有逻辑）
        for i, (class_id, x_center, y_center, width, height) in enumerate(annotations):
            x_center_abs = int(x_center * w)
            y_center_abs = int(y_center * h)
            width_abs = int(width * w)
            height_abs = int(height * h)
            
            x1 = int(x_center_abs - width_abs / 2)
            y1 = int(y_center_abs - height_abs / 2)
            x2 = int(x_center_abs + width_abs / 2)
            y2 = int(y_center_abs + height_abs / 2)
            
            x1 = max(0, min(w-1, x1))
            y1 = max(0, min(h-1, y1))
            x2 = max(0, min(w-1, x2))
            y2 = max(0, min(h-1, y2))
            
            color = self.colors[class_id % len(self.colors)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"Class {class_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(image, (x_center_abs, y_center_abs), 3, color, -1)
        
        # 保存或显示图像
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_dir / f"checked_{image_file}"
            cv2.imwrite(str(output_path), image)
            print(f"✅ 已保存检查结果：{output_path}")
        else:
            cv2.imshow(f"Label Check: {image_file}", image)
            print("按任意键继续，按'q'退出...")
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            return key == ord('q')
        
        return False
    
    def check_all_labels_format(self):
        """检查所有标签文件的格式（增强统计和提示）"""
        print("=" * 60)
        print("检查所有标签文件格式")
        print(f"图像目录：{self.image_dir}")
        print(f"标注目录：{self.label_dir}")
        print("=" * 60)
        
        # 获取所有图像文件（过滤有效扩展名）
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(valid_extensions)]
        
        total_errors = 0
        total_annotations = 0
        missing_labels = 0
        invalid_annotations = 0
        
        for image_file in image_files:
            label_filename = f"{os.path.splitext(image_file)[0]}.txt"
            label_path = self.label_dir / label_filename
            
            # 检查标注文件是否存在
            if not label_path.exists():
                print(f"❌ 标注文件缺失：{label_filename}（对应图像：{image_file}）")
                total_errors += 1
                missing_labels += 1
                continue
            
            # 读取并检查标注
            annotations = self.read_yolo_annotations(label_path)
            total_annotations += len(annotations)
            
            # 统计无效标注（文件存在但无有效标注）
            if len(annotations) == 0:
                print(f"⚠️  标注文件存在但无有效标注：{label_filename}（对应图像：{image_file}）")
                invalid_annotations += 1
                total_errors += 1
        
        # 输出汇总统计（新增详细分类）
        print("\n" + "=" * 60)
        print("检查结果汇总：")
        print(f"总图像数：{len(image_files)}")
        print(f"总标注文件数：{len(image_files) - missing_labels}（缺失 {missing_labels} 个）")
        print(f"总有效标注数：{total_annotations}")
        print(f"无效标注文件数（存在但无有效标注）：{invalid_annotations}")
        print(f"总错误数：{total_errors}")
        print("=" * 60)
        
        return total_errors == 0
    
    def analyze_class_distribution(self):
        """分析类别分布（保持原有逻辑，增强路径提示）"""
        print("=" * 60)
        print("分析类别分布")
        print(f"图像目录：{self.image_dir}")
        print(f"标注目录：{self.label_dir}")
        print("=" * 60)
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(valid_extensions)]
        
        class_counts = {}
        bbox_sizes = []
        skipped_files = 0
        
        for image_file in image_files:
            label_filename = f"{os.path.splitext(image_file)[0]}.txt"
            label_path = self.label_dir / label_filename
            
            if not label_path.exists():
                print(f"⚠️  跳过无标注的图像：{image_file}")
                skipped_files += 1
                continue
                
            annotations = self.read_yolo_annotations(label_path)
            if not annotations:
                print(f"⚠️  跳过无有效标注的文件：{label_filename}（对应图像：{image_file}）")
                skipped_files += 1
                continue
            
            for class_id, x_center, y_center, width, height in annotations:
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                bbox_sizes.append(width * height)
        
        print(f"\n处理完成：共分析 {len(image_files) - skipped_files} 个有效图像文件")
        
        # 打印类别分布
        if class_counts:
            print("\n类别分布:")
            total = sum(class_counts.values())
            for class_id in sorted(class_counts.keys()):
                count = class_counts[class_id]
                percentage = (count / total) * 100
                print(f"  类别 {class_id}: {count} 个 ({percentage:.1f}%)")
        else:
            print("\n未找到任何有效类别标注")
        
        # 打印边界框大小统计
        if bbox_sizes:
            avg_size = np.mean(bbox_sizes)
            min_size = np.min(bbox_sizes)
            max_size = np.max(bbox_sizes)
            print(f"\n边界框大小统计 (相对面积):")
            print(f"  平均大小: {avg_size:.4f}")
            print(f"  最小大小: {min_size:.4f}")
            print(f"  最大大小: {max_size:.4f}")
            
            small = len([s for s in bbox_sizes if s < 0.01])
            medium = len([s for s in bbox_sizes if 0.01 <= s < 0.1])
            large = len([s for s in bbox_sizes if s >= 0.1])
            
            total = len(bbox_sizes)
            print(f"  小目标 (<1%): {small} ({small/total*100:.1f}%)")
            print(f"  中目标 (1-10%): {medium} ({medium/total*100:.1f}%)")
            print(f"  大目标 (>=10%): {large} ({large/total*100:.1f}%)")
        
        return class_counts


def interactive_label_check():
    """交互式标签检查（修改路径为Path处理）"""
    # 注意：路径中的反斜杠请替换为正斜杠或双反斜杠，或使用原始字符串（在路径前加r）
    IMAGE_DIR = r"augmented_1000_fixed\images\train"  # 原始字符串避免转义问题
    LABEL_DIR = r"augmented_1000_fixed\labels\train"
    OUTPUT_DIR = "label_checks"
    
    try:
        checker = YOLOLabelChecker(IMAGE_DIR, LABEL_DIR)
    except FileNotFoundError as e:
        print(f"初始化失败：{e}")
        return
    
    print("YOLO标签检查工具")
    print("1. 检查所有标签格式")
    print("2. 分析类别分布")
    print("3. 可视化随机样本")
    print("4. 可视化特定图像")
    print("5. 批量检查并保存")
    
    choice = input("请选择检查方式 (1-5): ").strip()
    
    if choice == "1":
        checker.check_all_labels_format()
    
    elif choice == "2":
        checker.analyze_class_distribution()
    
    elif choice == "3":
        image_files = [f for f in os.listdir(checker.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print("未找到任何图像文件")
            return
            
        sample_size = min(10, len(image_files))
        samples = random.sample(image_files, sample_size)
        
        print(f"随机检查 {sample_size} 张图像:")
        for img_file in samples:
            if checker.visualize_single_image(img_file):
                break  # 用户按了'q'键
    
    elif choice == "4":
        image_files = [f for f in os.listdir(checker.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print("未找到任何图像文件")
            return
            
        print("可用图像（前20个）:")
        for i, img_file in enumerate(image_files[:20]):
            print(f"  {i+1}. {img_file}")
        
        try:
            idx = int(input("请输入图像编号: ")) - 1
            if 0 <= idx < len(image_files):
                checker.visualize_single_image(image_files[idx])
            else:
                print(f"编号超出范围（有效范围：1-{len(image_files)}）")
        except ValueError:
            print("请输入有效数字（如1、2）")
    
    elif choice == "5":
        image_files = [f for f in os.listdir(checker.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print("未找到任何图像文件")
            return
            
        sample_size = min(20, len(image_files))
        samples = random.sample(image_files, sample_size)
        
        print(f"批量检查 {sample_size} 张图像并保存到 {OUTPUT_DIR}...")
        for img_file in samples:
            checker.visualize_single_image(img_file, OUTPUT_DIR)
        
        print(f"检查结果已保存到: {OUTPUT_DIR}")
    
    else:
        print("无效选择，请输入1-5之间的数字")


if __name__ == "__main__":
    interactive_label_check()