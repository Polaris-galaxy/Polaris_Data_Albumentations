# Polaris Data Albumentations — 使用说明

本仓库包含基于 **Albumentations** 与自研逻辑的 **离线数据增强**、**YOLO 标注同步**、**格式转换与校验** 脚本。文件名在英文标识后常带 `_中文说明` 后缀；终端或资源管理器中引用时须写 **完整文件名**（含中文部分）。

---

## 1. 环境准备

1. 进入本仓库根目录（与 `requirements.txt` 同级）。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. **PyTorch**：若需 GPU，请到 [PyTorch 官网](https://pytorch.org/) 按本机 CUDA 版本单独安装；`requirements.txt` 中的 `torch` / `torchvision` 为版本下限说明。
4. 运行脚本时，请在命令中使用脚本相对仓库根目录的路径（见下文示例）。**Windows** 下建议在 PowerShell 或 CMD 中执行；路径含中文时保持与脚本内配置一致。

---

## 2. 目录与用途

| 目录 / 文件 | 用途概要 |
|-------------|----------|
| `picture_annotation_augmentation/` | **无检测标注** 的纯图像增强（如羊毛场景）。主脚本：`wool_image_albumentations_pipeline_羊毛图增强流水线.py`（颜色 / 几何 / 噪声 + 环境光照两阶段）。操作步骤见 **第 3 节**。 |
| `picture_yolo/` | **带 YOLO txt 标注** 的数据集：扩增到目标张数、训练前检查、目录诊断、标注可视化、YOLO→COCO、服务器 COCO 路径修正。 |
| `yolo_annotation_augmentation/` | **分步 YOLO 增强**（单图 / Mosaic / MixUp）与 **matplotlib 抽查**。 |
| `video_data_get/` | 从 **视频导出图片**：`extract_frames按比例抽帧.py` 按多视频总长比例分配目标帧数。操作步骤见 **第 4 节**。`数值按最小值归一演示.py` 为独立数值演示，与抽帧 / 增强流水线无关。 |

检测类任务常见数据流：准备 YOLO 格式 `images/` + `labels/` → 预检与扩增 → 可选 COCO 转换或 JSON 路径修正 → 送入 RT-DETR / YOLO 训练。  
数据来源为录像时，顺序可为：`video_data_get` 抽帧 → 标注 → `picture_yolo` / `yolo_annotation_augmentation`。

---

## 3. `picture_annotation_augmentation`：无标注图片（羊毛两阶段增强）

**目录**：`picture_annotation_augmentation/`（仅含本流水线脚本，无 YOLO 标注输出）。

**脚本**：`picture_annotation_augmentation/wool_image_albumentations_pipeline_羊毛图增强流水线.py`

1. 打开文件，在末尾 `if __name__ == "__main__":` 中修改：
   - **第一次**：`process_wool_images(...)` 的 `input_folder`、`output_folder`、`augmentation_strength`（`'light'` / `'moderate'` / `'heavy'`）、`target_multiplier`、`wool_type`（如 `'fine'` / `'coarse'` 或留空）。
   - **第二次（环境类）**：`second_environment_process(...)` 的 `input_folder`（一般为第一次输出目录）、`output_folder`、`target_multiplier`。
2. 在仓库根目录执行：

```bash
python picture_annotation_augmentation/wool_image_albumentations_pipeline_羊毛图增强流水线.py
```

说明：第一次侧重颜色、几何与噪声等；第二次侧重环境、光照（如雨效等）。仅生成图片，**不含**检测框标注。

---

## 4. `video_data_get`：从视频按比例抽帧

**目录**：`video_data_get/`

**脚本**：`video_data_get/extract_frames按比例抽帧.py`

1. 打开文件中的 `video_to_frames()`，修改：
   - `VIDEO_PATH`：存放 **`.avi` 视频** 的文件夹（脚本当前按后缀筛选 `.avi`，若使用 mp4 等需在脚本内把筛选条件改为包含对应后缀）。
   - `OUTPUT_DIR`：抽帧图片保存目录（不存在会自动创建）。
   - `TARGET_IMAGE_NUMBER`：所有视频合计要抽取的 **总帧数**；脚本会按 **各视频总帧数占全部视频总帧数的比例** 分配每个视频抽多少帧，再在各自时长内 **均匀取帧**。
2. 在仓库根目录执行：

```bash
python video_data_get/extract_frames按比例抽帧.py
```

依赖 **OpenCV**（`opencv-python`，见 `requirements.txt`）。抽得的图片可用于人工标注或再接入第 5 节及以后的 YOLO 增强流程。

**其他**：`video_data_get/数值按最小值归一演示.py` 为列表归一化的小演示代码，不参与抽帧或增强流水线，无需随项目常规使用。

---

## 5. YOLO 数据集：扩增到目标张数（主流程）

**脚本**：`picture_yolo/yolo_albumentations增强至目标张数.py`

1. 数据目录要求：图像与 YOLO 标签分别放在两个文件夹中（常见为 `.../images`、`.../labels`，与 `FixedYOLOAugmentor` 构造参数一致）。
2. 打开脚本，在 `main()` 中修改：
   - `IMAGE_DIR`、`LABEL_DIR`、`OUTPUT_DIR`：输入图、输入标签、**输出根目录**（脚本会在其下创建 `images/`、`labels/`）。
   - `VALID_CLASS_IDS`：单类检测示例 `[0]`；多类写 `[0,1,...]`；不校验类别用 `None`（勿误写为整数 `0`）。
   - `TARGET_COUNT`、`MIN_BBOX_AREA`、`PROMINENT_MIN_AREA` 等：按小目标与数据质量调整（脚本内注释有说明）。
3. 运行：

```bash
python picture_yolo/yolo_albumentations增强至目标张数.py
```

输出为增强后的 YOLO 数据集，可用于后续训练。

---

## 6. 分步 YOLO 增强（单图 + Mosaic + MixUp）

**脚本**：`yolo_annotation_augmentation/run_yolo_stepwise_augmentation_分步YOLO增强入口.py`

1. 打开文件，在 `step_by_step_augmentation()` 内修改 `dataset_dir`（需含 `images/`、`labels/`）与 `output_dir`。
2. 在 **包含该脚本的目录** 下运行，以便正确导入同目录下的 `yolo_augmentation_library_增强核心库.py`：

```bash
cd yolo_annotation_augmentation
python run_yolo_stepwise_augmentation_分步YOLO增强入口.py
```

（若从仓库根目录运行，需自行配置 `PYTHONPATH` 或改为包导入，一般推荐 `cd` 后执行。）

---

## 7. 增强结果可视化抽查

**脚本**：`yolo_annotation_augmentation/verify_yolo_augmentation_matplotlib_增强效果可视化验证.py`

1. 打开文件，在 `if __name__ == "__main__":` 中将 `verify_augmentation('...')` 的参数改为你的数据集根目录（其下应有 `images/`、`labels/`）。
2. 运行：

```bash
python yolo_annotation_augmentation/verify_yolo_augmentation_matplotlib_增强效果可视化验证.py
```

在当前工作目录生成 `augmentation_verification.jpg`；脚本含 `plt.show()`，仅在有图形界面的环境下会显示窗口。

---

## 8. 训练前检查与辅助工具

| 脚本 | 作用与用法要点 |
|------|----------------|
| `picture_yolo/训练前图与标签检查.py` | 检查 `original_data/images` 与 `original_data/labels` 是否成对；可在脚本内改路径常量后运行：`python picture_yolo/训练前图与标签检查.py` |
| `picture_yolo/数据集目录诊断.py` | 扫描当前工作目录下常见文件夹名，辅助排查路径问题；在目标目录旁执行或按需改脚本内逻辑。 |
| `picture_yolo/yolo_label_checker_YOLO标注可视化检查.py` | 交互式把 YOLO 框画在图上；在 `interactive_label_check()` 内修改 `IMAGE_DIR`、`LABEL_DIR`、`OUTPUT_DIR` 后运行：`python picture_yolo/yolo_label_checker_YOLO标注可视化检查.py` |

---

## 9. YOLO 转 COCO（命令行）

**脚本**：`picture_yolo/YOLO转COCO格式.py`

数据集根目录建议采用 **按划分分子目录** 的结构，例如：

- `images/train`、`images/val`（可选 `images/test`）
- `labels/train`、`labels/val`（与上面划分一一对应）

根目录可放 `data.yaml`（含 `names`、`nc` 等）以便类别名与校验。若结构不同，需先整理为上述形式或按需改转换脚本。

```bash
python picture_yolo/YOLO转COCO格式.py --yolo_path "你的YOLO数据集根目录" --output_dir "COCO输出目录"
```

需要把图片一并复制到输出目录时，加 `--copy-images`。

---

## 10. 服务器训练：修正 COCO 中图片路径

**脚本**：`picture_yolo/coco_json_fix_image_paths_for_server_COCO路径修正服务器.py`

在 **Linux 训练机** 上按需修改文件顶部 `ANNOTATION_PATH`、`IMAGE_ABS_DIR`（COCO JSON 与图片真实绝对路径），保存后执行：

```bash
python picture_yolo/coco_json_fix_image_paths_for_server_COCO路径修正服务器.py
```

脚本会备份原 JSON，再将 `images` 中路径改写为与服务器目录一致。细则见同目录 `服务器训练COCO指引.md`（若存在）。

---

## 11. 与 RT-DETR 训练工程的数据衔接

增强与格式整理在本仓库输出；在 **RT-DETR** 等工程中，把数据 YAML 里 `train` / `val` 指到增强后的图片目录（或 COCO 标注与图片路径）。RT-DETR 侧数据集字段说明见该项目 README。

**本仓库 Git 地址**：<https://github.com/Polaris-galaxy/Polaris_Data_Albumentations.git>
