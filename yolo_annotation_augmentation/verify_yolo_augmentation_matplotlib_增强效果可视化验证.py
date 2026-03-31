import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def verify_augmentation(dataset_dir):
    """验证增强结果是否正确"""
    
    images_dir = Path(dataset_dir) / 'images'
    labels_dir = Path(dataset_dir) / 'labels'
    
    # 获取前5张增强图像进行验证
    image_files = list(images_dir.glob('*aug*.jpg'))[:5]
    image_files.extend(list(images_dir.glob('*mosaic*.jpg'))[:2])
    image_files.extend(list(images_dir.glob('*mixup*.jpg'))[:2])
    
    if not image_files:
        print("未找到增强图像，请检查输出目录")
        return
    
    # 创建验证图
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, img_path in enumerate(image_files[:9]):  # 最多显示9张
        if i >= 9:
            break
            
        # 读取图像
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        
        # 查找对应的标注文件
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        # 显示图像
        axes[i].imshow(image)
        axes[i].set_title(f'{img_path.name}')
        axes[i].axis('off')
        
        # 显示边界框
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                data = line.strip().split()
                if len(data) == 5:
                    class_id, x_center, y_center, width, height = map(float, data)
                    
                    # 转换为像素坐标
                    img_height, img_width = image.shape[:2]
                    x = (x_center - width/2) * img_width
                    y = (y_center - height/2) * img_height
                    w = width * img_width
                    h = height * img_height
                    
                    # 添加边界框
                    rect = Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor='red', facecolor='none')
                    axes[i].add_patch(rect)
                    
                    # 添加类别标签
                    axes[i].text(x, y-5, f'Class {int(class_id)}', 
                               color='red', fontsize=10, weight='bold')
    
    # 隐藏多余的子图
    for i in range(len(image_files), 9):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_verification.jpg', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"验证完成！结果已保存为 'augmentation_verification.jpg'")
    print(f"共增强图像数量: {len(list(images_dir.glob('*.jpg')))}")
    print(f"共增强标注数量: {len(list(labels_dir.glob('*.txt')))}")

if __name__ == "__main__":
    verify_augmentation('d:\Galaxy\其他\桌面\yolo_data\step_augmented_dataset')  # 替换为您的增强数据集路径