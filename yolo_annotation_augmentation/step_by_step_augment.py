import cv2
import os
from pathlib import Path
from yolo_augmentation import YOLOAugmentor, MosaicAugmentor, MixUpAugmentor

def step_by_step_augmentation():
    """分步进行数据增强，提供更精细的控制"""
    
    # 数据集路径
    dataset_dir = 'D:/Galaxy/其他/桌面/yolo_data'
    output_dir = 'D:/Galaxy/其他/桌面/yolo_data/step_augmented_dataset'
    
    # 创建输出目录
    os.makedirs(f'{output_dir}/images', exist_ok=True)
    os.makedirs(f'{output_dir}/labels', exist_ok=True)
    
    # 初始化增强器
    single_augmentor = YOLOAugmentor(image_size=640)
    mosaic_augmentor = MosaicAugmentor(image_size=640)
    mixup_augmentor = MixUpAugmentor(image_size=640)
    
    # 获取所有图像文件
    image_files = list(Path(f'{dataset_dir}/images').glob('*.jpg'))
    image_files.extend(Path(f'{dataset_dir}/images').glob('*.png'))
    image_files.extend(Path(f'{dataset_dir}/images').glob('*.jpeg'))
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 复制原始数据
    print("复制原始数据...")
    for img_path in image_files:
        # 复制图像
        ann_path = Path(f'{dataset_dir}/labels/{img_path.stem}.txt')
        if ann_path.exists():
            # 复制到输出目录
            cv2.imwrite(f'{output_dir}/images/{img_path.name}', cv2.imread(str(img_path)))
            single_augmentor.save_yolo_annotation(
                *single_augmentor.parse_yolo_annotation(str(ann_path)),
                f'{output_dir}/labels/{img_path.stem}.txt'
            )
    
    # 单图增强
    print("进行单图增强...")
    augmentation_count = 0
    for img_path in image_files:
        ann_path = Path(f'{dataset_dir}/labels/{img_path.stem}.txt')
        
        if not ann_path.exists():
            continue
            
        # 为每张图像生成2个增强版本
        for i in range(2):
            try:
                augmented_image, bboxes, labels = single_augmentor.augment_single_image(
                    str(img_path), str(ann_path)
                )
                
                # 保存增强结果
                aug_name = f"{img_path.stem}_single_aug_{i}{img_path.suffix}"
                cv2.imwrite(f'{output_dir}/images/{aug_name}', 
                           cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
                single_augmentor.save_yolo_annotation(
                    bboxes, labels, f'{output_dir}/labels/{img_path.stem}_single_aug_{i}.txt'
                )
                
                augmentation_count += 1
            except Exception as e:
                print(f"单图增强失败 {img_path}: {e}")
    
    # Mosaic增强（需要至少4张图像）
    print("进行Mosaic增强...")
    if len(image_files) >= 4:
        mosaic_count = min(10, len(image_files) // 4)  # 最多生成10个mosaic
        for i in range(mosaic_count):
            try:
                # 随机选择4张图像
                selected_images = []
                selected_annotations = []
                
                for j in range(4):
                    img_idx = (i * 4 + j) % len(image_files)
                    img_path = image_files[img_idx]
                    ann_path = Path(f'{dataset_dir}/labels/{img_path.stem}.txt')
                    
                    if ann_path.exists():
                        selected_images.append(str(img_path))
                        selected_annotations.append(str(ann_path))
                
                if len(selected_images) == 4:
                    mosaic_image, bboxes, labels = mosaic_augmentor.create_mosaic(
                        selected_images, selected_annotations
                    )
                    
                    # 保存mosaic结果
                    mosaic_name = f"mosaic_{i}.jpg"
                    cv2.imwrite(f'{output_dir}/images/{mosaic_name}', 
                               cv2.cvtColor(mosaic_image, cv2.COLOR_RGB2BGR))
                    single_augmentor.save_yolo_annotation(
                        bboxes, labels, f'{output_dir}/labels/mosaic_{i}.txt'
                    )
                    
                    augmentation_count += 1
            except Exception as e:
                print(f"Mosaic增强失败: {e}")
    
    print(f"增强完成！共生成 {augmentation_count} 张增强图像")

if __name__ == "__main__":
    step_by_step_augmentation()