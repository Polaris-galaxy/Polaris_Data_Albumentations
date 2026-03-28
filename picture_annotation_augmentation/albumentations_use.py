# import warnings
# warnings.filterwarnings("ignore", message="A new version of Albumentations is available")
# import os
# import albumentations as A
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# def augment_images_multiple_times(input_folder, output_folder, multiplier=5, augmentation_type='moderate'):
#     """
#     对输入文件夹中的所有图片进行多倍增强
    
#     参数:
#     - input_folder: 输入图片文件夹路径
#     - output_folder: 输出增强图片文件夹路径
#     - multiplier: 每张原始图片生成的增强图片数量
#     - augmentation_type: 增强类型 ('light', 'moderate', 'heavy')
#     """
#     # 创建输出文件夹
#     os.makedirs(output_folder, exist_ok=True)
    
#     # 定义增强管道
#     if augmentation_type == 'light':
#         transform = A.Compose([
#             A.HorizontalFlip(p=0.3),           # 水平翻转，概率30%
#             A.RandomBrightnessContrast(p=0.2), # 随机亮度和对比度调整，概率20%
#             A.GaussianBlur(p=0.1),             # 高斯模糊，概率10%
#         ])
#     elif augmentation_type == 'moderate':
#         transform = A.Compose([
#            A.HorizontalFlip(p=0.5),           # 水平翻转，概率50%
#             A.OneOf([                          # 从以下两种中选择一种
#                 A.RandomBrightnessContrast(p=1.0),  # 随机亮度和对比度调整
#                 A.HueSaturationValue(p=1.0),        # 随机色调、饱和度和明度调整
#             ], p=0.3),                         # 这个OneOf块执行的概率为30%
#             A.GaussianBlur(p=0.2),             # 高斯模糊，概率20%
#         ])
#     elif augmentation_type == 'heavy':
#         transform = A.Compose([
#             A.HorizontalFlip(p=0.7),           # 水平翻转，概率70%
#             A.RandomBrightnessContrast(p=0.6), # 随机亮度和对比度调整，概率60%
#             A.OneOf([                          # 从以下三种中选择一种
#                 A.GaussianBlur(p=1.0),         # 高斯模糊
#                 A.MedianBlur(p=1.0),           # 中值模糊
#                 A.MotionBlur(p=1.0),           # 运动模糊
#             ], p=0.4),                         # 这个OneOf块执行的概率为40%
#         ])
    
#     elif augmentation_type == 'environment' : 
#         transform = A.Compose([
#             A.RandomRain(p=0.5),
#             A.RandomSnow(p=0.5) 
#         ])
#     # 获取所有图片文件
#     image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
#     image_files = [f for f in os.listdir(input_folder) 
#                   if os.path.splitext(f)[1].lower() in image_extensions]
    
#     print(f"找到 {len(image_files)} 张图片，将生成 {len(image_files) * multiplier} 张增强图片")
    
#     # 处理每张图片
#     for img_file in image_files:
#         img_path = os.path.join(input_folder, img_file)
#         image = cv2.imread(img_path)
        
#         if image is None:
#             print(f"无法读取图片: {img_file}")
#             continue
            
#         # 转换为RGB (OpenCV默认是BGR)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # 生成多个增强版本
#         base_name = os.path.splitext(img_file)[0]
        
#         for i in range(multiplier):
#             # 应用增强
#             augmented = transform(image=image)
#             augmented_image = augmented['image']
            
#             # 保存增强后的图片
#             output_filename = f"{base_name}_aug_{i+1}.jpg"
#             output_path = os.path.join(output_folder, output_filename)
            
#             # 转换回BGR以便OpenCV保存
#             save_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
#             cv2.imwrite(output_path, save_image)
    
#     print("增强完成!")

# augment_images_multiple_times(
#     input_folder="D:/Galaxy/其他/桌面/数据增强",            # 输入路径
#     output_folder="D:/Galaxy/其他/桌面/数据输出（重度）",      # 输出路径
#     multiplier=5,  # 每张原始图片生成5张增强图片
#     augmentation_type='environment'                   # 增强模式
# )
import warnings
warnings.filterwarnings("ignore", message="A new version of Albumentations is available")
import albumentations as A
import os
import cv2
import numpy as np
from pathlib import Path

def wool_specific_augmentation():
    """
    针对羊毛图像特性的数据增强管道
    """
    # 轻度增强 - 保持羊毛纹理特性
    light_wool_transform = A.Compose([
        # 颜色增强 - 羊毛颜色变化很重要
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.3),
        
        # 几何变换 - 保持羊毛纤维结构
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        
        # 轻微模糊 - 模拟不同焦距
        A.GaussianBlur(blur_limit=3, p=0.2),
    ])
    
    # 中度增强 - 增加多样性但保持可识别性
    moderate_wool_transform = A.Compose([
        # 更强的颜色变换
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
        A.CLAHE(clip_limit=2.0, p=0.3),
        
        # 几何变换
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15, 
            border_mode=cv2.BORDER_REFLECT, p=0.5
        ),
        
        # 纹理相关增强
        A.GaussianBlur(blur_limit=5, p=0.3),
        A.MedianBlur(blur_limit=3, p=0.2),
        
        # 噪声增强 - 模拟不同拍摄条件
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    ])
    
    # 重度增强 - 用于提高模型鲁棒性
    heavy_wool_transform = A.Compose([
        # 强烈的颜色变换
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.4),
        
        # 复杂的几何变换
        A.HorizontalFlip(p=0.7),
        A.VerticalFlip(p=0.7),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=30, 
            border_mode=cv2.BORDER_REFLECT, p=0.6
        ),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        
        # 多种模糊效果
        A.OneOf([
            A.GaussianBlur(blur_limit=7, p=0.5),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.MedianBlur(blur_limit=5, p=0.2),
        ], p=0.4),
        
        # 噪声和压缩伪影
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 100.0), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            A.ImageCompression(quality_lower=60, quality_upper=90, p=0.2),
        ] ,p=0.3),
        
        # 模拟不同光照条件
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    ])
    
    return {
        'light': light_wool_transform,
        'moderate': moderate_wool_transform,
        'heavy': heavy_wool_transform
    }

def wool_fiber_aware_augmentation():
    """
    考虑羊毛纤维方向的增强策略
    避免过度破坏纤维结构特征
    """
    transform = A.Compose([
        # 保持纤维结构的旋转
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.4),
        
        # 小幅度的仿射变换
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(0.05, 0.05),
            rotate=(-10, 10),
            shear=(-5, 5),
            p=0.5
        ),
        
        # 颜色增强 - 不改变纤维结构
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=5, p=0.3),
        
        # 纹理增强
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.MedianBlur(blur_limit=3, p=0.2),
        
        # 噪声增强
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
    ])
    
    return transform

def wool_quality_augmentation():
    """
    针对羊毛质量检测的增强策略
    重点增强与质量相关的特征
    """
    transform = A.Compose([
        # 增强纹理特征
        A.OneOf([
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.4),
            A.Emboss(alpha=(0.1, 0.3), strength=(0.5, 1.0), p=0.3),
            A.RandomToneCurve(scale=0.1, p=0.3),
        ], p=0.5),
        
        # 颜色增强 - 羊毛颜色是重要质量指标
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
            A.ChannelShuffle(p=0.2),
        ], p=0.6),
        
        # 几何变换 - 不同角度观察羊毛
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.15, rotate_limit=45,
            border_mode=cv2.BORDER_REFLECT, p=0.5
        ),
        
        # 模拟不同拍摄条件
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=0.4),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
        ], p=0.4),
        
        # 增强对比度以突出纤维结构
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.5),
        A.CLAHE(clip_limit=3.0, p=0.3),
    ])
    
    return transform

def wool_environment_augmentation():

    """
    针对羊毛图像在不同环境下的增强策略
    模拟不同光照和天气条件
    """
    transform = A.Compose([
        # 模拟不同光照条件
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        
        # 模拟雨天或雪天效果
        A.RandomRain(p=0.3),
        # A.RandomSnow(p=0.1),
        
        # 颜色增强 - 保持羊毛颜色特性
        # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
        
        # 几何变换 - 保持羊毛结构
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # 噪声增强 - 模拟不同拍摄条件
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.015),
    ])
    
    return transform

def process_wool_images(input_folder, output_folder, augmentation_strength='moderate', 
                       target_multiplier=5, wool_type=None):
    """
    完整的羊毛图像处理和数据增强流程
    
    参数:
    - input_folder: 输入图像文件夹
    - output_folder: 输出文件夹
    - augmentation_strength: 增强强度 ('light', 'moderate', 'heavy')
    - target_multiplier: 目标增强倍数
    - wool_type: 羊毛类型（可选，用于特定调整）
    """
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 根据羊毛类型选择增强策略
    if wool_type == 'fine':
        # 细羊毛 - 更注重纹理保持
        transform = wool_fiber_aware_augmentation()
    elif wool_type == 'coarse':
        # 粗羊毛 - 可以接受更强的增强
        wool_transforms = wool_specific_augmentation()
        transform = wool_transforms.get(augmentation_strength, wool_transforms['moderate'])
    else:
        # 通用羊毛增强
        wool_transforms = wool_specific_augmentation()
        transform = wool_transforms.get(augmentation_strength, wool_transforms['moderate'])
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(input_folder) 
                  if Path(f).suffix.lower() in image_extensions]
    
    print(f"找到 {len(image_files)} 张羊毛图像")
    print(f"增强强度: {augmentation_strength}")
    print(f"目标增强倍数: {target_multiplier}")
    
    # 计算每张图像需要生成的增强版本数量
    images_needed = target_multiplier
    if len(image_files) > 0:
        images_per_original = max(1, images_needed // len(image_files))
    else:
        print("未找到图像文件")
        return
    
    # 处理每张图像
    total_generated = 0
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"无法读取图像: {img_file}")
            continue
            
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        base_name = Path(img_file).stem
        
        # 生成增强版本
        for i in range(images_per_original):
            try:
                # 应用增强
                augmented = transform(image=image_rgb)
                augmented_image = augmented['image']
                
                # 保存增强后的图像
                output_filename = f"{base_name}_wool_aug_{i+1}.jpg"
                output_path = os.path.join(output_folder, output_filename)
                
                # 转换回BGR保存
                save_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, save_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                total_generated += 1
                
            except Exception as e:
                print(f"增强 {img_file} 时出错: {str(e)}")
                continue
    
    print(f"增强完成! 共生成 {total_generated} 张增强图像")
    print(f"原始图像数量: {len(image_files)}")
    print(f"平均每张原始图像生成: {total_generated/len(image_files):.1f} 张增强图像")

def second_environment_process(input_folder, output_folder, target_multiplier=5):
    """
    第二次环境处理，进一步增强图像
    参数:
    - input_folder: 输入图像文件夹
    - output_folder: 输出文件夹
    - target_multiplier: 目标增强倍数
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    transform = wool_environment_augmentation()
     # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(input_folder) 
                  if Path(f).suffix.lower() in image_extensions]
    
    print(f"找到 {len(image_files)} 张羊毛图像")
    print(f"增强强度: environment")
    print(f"目标增强倍数: {target_multiplier}")

     # 计算每张图像需要生成的增强版本数量
    images_needed = target_multiplier
    if len(image_files) > 0:
        images_per_original = max(1, images_needed // len(image_files))
    else:
        print("未找到图像文件")
        return
    
    # 处理每张图像
    total_generated = 0
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"无法读取图像: {img_file}")
            continue
            
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        base_name = Path(img_file).stem
        
        # 生成增强版本
        for i in range(images_per_original):
            try:
                # 应用增强
                augmented = transform(image=image_rgb)
                augmented_image = augmented['image']
                
                # 保存增强后的图像
                output_filename = f"{base_name}_wool_aug_{i+1}.jpg"
                output_path = os.path.join(output_folder, output_filename)
                
                # 转换回BGR保存
                save_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, save_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                total_generated += 1
                
            except Exception as e:
                print(f"增强 {img_file} 时出错: {str(e)}")
                continue
    
    print(f"增强完成! 共生成 {total_generated} 张增强图像")
    print(f"原始图像数量: {len(image_files)}")
    print(f"平均每张原始图像生成: {total_generated/len(image_files):.1f} 张增强图像")

    transform = wool_environment_augmentation()
    return transform
# 使用示例
if __name__ == "__main__":
    process_wool_images(
        input_folder="D:/Galaxy/其他/桌面/数据增强",
        output_folder="D:/Galaxy/其他/桌面/数据输出（重度）",
        augmentation_strength='heavy',  # 可选 'light', 'moderate', 'heavy'
        target_multiplier=100,
        wool_type=''  
    )

    print("第一次增强完成，开始第二次环境增强...")
    transform = second_environment_process(input_folder="D:/Galaxy/其他/桌面/数据输出（重度）",
                                           output_folder="D:/Galaxy/其他/桌面/二次增强（重度）",
                                            target_multiplier=600
                                            )