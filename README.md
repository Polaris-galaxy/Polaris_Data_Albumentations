# 🚀快速开始

> 说明：各脚本文件名在英文标识后带有 `_中文说明`，便于从资源管理器或终端一眼看出用途；运行命令时请使用**完整文件名**（含中文部分）。


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

## 支持效果（运行 `yolo_annotation_augmentation/run_yolo_stepwise_augmentation_分步YOLO增强入口.py`）

✅ 支持YOLO格式数据集

✅ 单图增强（几何变换、颜色变换、噪声添加等）

✅ Mosaic增强（4图拼接）

✅ MixUp增强（2图混合）

✅ 自动处理标注文件

✅ 可视化验证增强效果

✅ 灵活的参数配置

## 验证（运行 `yolo_annotation_augmentation/verify_yolo_augmentation_matplotlib_增强效果可视化验证.py`）

# 少数图片标注直接生成大量增强数据集

## 🚀使用说明

使用 `picture_yolo/yolo_images_labels_preflight_check_训练前图与标签检查.py` 做训练前数据预检

使用 `picture_yolo/yolo_albumentations_augment_to_target_YOLO增强至目标张数.py` 进行 YOLO 目标张数增强流水线

使用路径诊断与标注可视化工具检查增强是否合理（见 `picture_yolo/diagnose_dataset_folder_structure_数据集目录诊断.py`、`picture_yolo/yolo_label_visual_checker_YOLO标注可视化检查.py`）

使用 `picture_yolo/convert_yolo_dataset_to_coco_YOLO转COCO格式.py` 将 YOLO 数据集转为 COCO

`picture_yolo/coco_json_fix_image_paths_for_server_COCO路径修正服务器.py` 用于服务器（Linux）训练时修正 COCO 中图片路径，按实际路径修改配置后运行即可。