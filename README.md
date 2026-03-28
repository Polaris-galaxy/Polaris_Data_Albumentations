# 🚀快速开始

# 图片类型数据增强

## 🌟使用说明

### 参数设置

#### 第一次增强参数

🌟- input_folder: 输入图像文件夹

🌟- output_folder: 输出文件夹

🌟- augmentation_strength: 增强强度 ('light', 'moderate', 'heavy')

🌟- target_multiplier: 目标增强倍数

🌟- wool_type: 羊毛类型（可选，用于特定调整）

#### 第二次增强参数

- input_folder: 输入图像文件夹

- output_folder: 输出文件夹

- target_multiplier: 目标增强倍数

### 📋增强方向

#### 第一次增强

✅颜色变化

✅物理拉伸

✅噪音增强等

#### 第二次增强

✅环境变化

✅光照变化等

# yolo数据集增强及验证

## 支持效果（运行step_by_step_augment.py）

✅ 支持YOLO格式数据集

✅ 单图增强（几何变换、颜色变换、噪声添加等）

✅ Mosaic增强（4图拼接）

✅ MixUp增强（2图混合）

✅ 自动处理标注文件

✅ 可视化验证增强效果

✅ 灵活的参数配置

## 验证（运行verify_augmentation(test).py）

# 少数图片标注直接生成大量增强数据集

## 🚀使用说明

使用 simple_debug.py 先进训练前检测

使用修复后的代码 fixed_augmentation.py 开始进行数据增强

使用路径及检测工具检测增强是否合理

在使用转化脚本将yolo格式的数据集转化为coco数据集

（AutoDL_coco_fix.py）用于服务器训练时在linux下路径问题，使用时按照代码酌情修改即可。