import os
import cv2
import numpy as np

def simple_data_check():
    """简单数据检查"""
    IMAGE_DIR = "original_data/images"
    LABEL_DIR = "original_data/labels"
    
    print("=" * 50)
    print("简单数据检查")
    print("=" * 50)
    
    # 检查目录是否存在
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ 图片目录不存在: {IMAGE_DIR}")
        return False
        
    if not os.path.exists(LABEL_DIR):
        print(f"❌ 标签目录不存在: {LABEL_DIR}")
        return False
    
    # 获取图片文件
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"找到 {len(image_files)} 张图片")
    
    if len(image_files) == 0:
        print("❌ 没有找到任何图片文件")
        return False
    
    # 检查前5张图片
    print("\n检查前5张图片:")
    for i in range(min(5, len(image_files))):
        img_path = os.path.join(IMAGE_DIR, image_files[i])
        label_file = image_files[i].replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(LABEL_DIR, label_file)
        
        # 检查图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ❌ {image_files[i]}: 无法读取图片")
        else:
            print(f"  ✅ {image_files[i]}: 尺寸 {img.shape}")
        
        # 检查标签
        if not os.path.exists(label_path):
            print(f"  ❌ {label_file}: 标签文件不存在")
        else:
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    print(f"  ✅ {label_file}: {len(lines)} 个标注")
                    # 显示第一个标注
                    if lines:
                        print(f"      示例: {lines[0].strip()}")
            except Exception as e:
                print(f"  ❌ {label_file}: 读取错误 - {e}")
    
    return True

def test_simple_augmentation():
    """测试简单的增强"""
    print("\n" + "=" * 50)
    print("测试简单增强")
    print("=" * 50)
    
    # 使用第一张图片测试
    IMAGE_DIR = "original_data/images"
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        return
        
    test_image = image_files[0]
    test_image_path = os.path.join(IMAGE_DIR, test_image)
    
    # 读取图片
    img = cv2.imread(test_image_path)
    if img is None:
        print("❌ 无法读取测试图片")
        return
    
    print(f"测试图片: {test_image}")
    print(f"原始尺寸: {img.shape}")
    
    # 简单的水平翻转
    flipped = cv2.flip(img, 1)  # 水平翻转
    
    # 保存测试结果
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(output_dir, "original.jpg"), img)
    cv2.imwrite(os.path.join(output_dir, "flipped.jpg"), flipped)
    
    print(f"✅ 增强测试完成，结果保存在 {output_dir} 目录")
    print("   查看 original.jpg 和 flipped.jpg 确认增强效果")

if __name__ == "__main__":
    if simple_data_check():
        test_simple_augmentation()