import json
import os
import shutil
from datetime import datetime


# 配置（务必确认图片实际存放目录！）
# 注意：Linux 下路径需以 / 开头，且要匹配图片实际位置（当前用户是 root，路径大概率带 /root/）
ANNOTATION_PATH = "/root/Deformable-DETR/my_yangmao_dataset_yolo/annotations_2/instances_train.json"
# 请根据实际图片存放目录修改！示例：如果图片在 /root/.../annotations_2/images/ 下，就写这个路径
IMAGE_ABS_DIR = "/root/Deformable-DETR/my_yangmao_dataset_yolo/annotations_2/images/"


def extract_pure_filename(file_path: str) -> str:
    """
    强制提取纯文件名（兼容 Windows/Linux 路径格式）
    无论输入是完整路径还是仅文件名，都返回最后一个分隔符后的部分
    """
    # 1. 统一替换 Windows 分隔符 \ 为 Linux 分隔符 /
    file_path = file_path.replace("\\", "/")
    # 2. 按 / 分割，过滤空字符串（避免路径末尾有 / 的情况）
    path_parts = [part for part in file_path.split("/") if part.strip()]
    # 3. 返回最后一部分（纯文件名）
    return path_parts[-1] if path_parts else ""


def main():
    # ---------------------- 提前检查关键目录/文件 ----------------------
    if not os.path.exists(ANNOTATION_PATH):
        print(f"❌ 错误：原标注文件 {ANNOTATION_PATH} 不存在！")
        return

    if not os.path.exists(IMAGE_ABS_DIR):
        print(f"❌ 错误：图片目录 {IMAGE_ABS_DIR} 不存在！请修正 IMAGE_ABS_DIR 配置")
        return

    # ---------------------- 备份逻辑 ----------------------
    ORIGINAL_BACKUP_PATH = f"{ANNOTATION_PATH}.original_backup"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    MODIFY_BACKUP_PATH = f"{ANNOTATION_PATH}.modify_{timestamp}_backup"

    try:
        if not os.path.exists(ORIGINAL_BACKUP_PATH):
            shutil.copy2(ANNOTATION_PATH, ORIGINAL_BACKUP_PATH)
            print(f"✅ 已备份原始文件到：{ORIGINAL_BACKUP_PATH}")
        else:
            print(f"ℹ️  原始文件备份已存在（{ORIGINAL_BACKUP_PATH}）")

        shutil.copy2(ANNOTATION_PATH, MODIFY_BACKUP_PATH)
        print(f"✅ 已备份当前文件到：{MODIFY_BACKUP_PATH}")

    except Exception as e:
        print(f"❌ 备份失败：{str(e)}")
        return

    # ---------------------- 读取标注数据 ----------------------
    try:
        with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ 错误：{ANNOTATION_PATH} 不是有效JSON！")
        return
    except Exception as e:
        print(f"❌ 读取失败：{str(e)}")
        return

    # ---------------------- 修正路径（核心优化：强制提取纯文件名） ----------------------
    fixed_count = 0
    empty_file_name_count = 0
    already_abs_path_count = 0

    if "images" not in coco_data:
        print("❌ 错误：未找到 'images' 字段！")
        return

    for img in coco_data["images"]:
        old_path = img.get("file_name", "").strip()
        
        if not old_path:
            empty_file_name_count += 1
            continue

        # 跳过已为有效绝对路径的情况（避免重复处理）
        if os.path.isabs(old_path) and os.path.exists(old_path):
            already_abs_path_count += 1
            continue

        # 核心修复：强制提取纯文件名（兼容 Windows 路径）
        pure_img_name = extract_pure_filename(old_path)
        if not pure_img_name:  # 极端情况：路径无效，无法提取文件名
            empty_file_name_count += 1
            continue

        # 拼接 Linux 绝对路径
        img_abs_path = os.path.join(IMAGE_ABS_DIR, pure_img_name)
        img["file_name"] = img_abs_path
        fixed_count += 1

    # ---------------------- 保存文件 ----------------------
    try:
        with open(ANNOTATION_PATH, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"❌ 写入失败：{str(e)}")
        return

    # ---------------------- 结果统计与验证 ----------------------
    total_images = len(coco_data["images"])
    print(f"\n=== 路径修正结果 ===")
    print(f"总图片数量：{total_images}")
    print(f"✅ 成功修正路径数：{fixed_count}")
    print(f"ℹ️  已为有效绝对路径数：{already_abs_path_count}")
    print(f"⚠️  无效路径/空file_name数：{empty_file_name_count}")

    if not coco_data["images"]:
        print("❌ 警告：'images' 列表为空！")
        return

    # 抽样验证（显示提取后的纯文件名和最终路径）
    sample_imgs = coco_data["images"][:3]
    exist_count = 0
    print(f"\n=== 抽样验证（纯文件名+最终路径）===")
    for idx, img in enumerate(sample_imgs, 1):
        pure_name = extract_pure_filename(img["file_name"])
        final_path = img["file_name"]
        exists = os.path.exists(final_path)
        if exists:
            exist_count += 1
        status = "✅ 存在" if exists else "❌ 不存在"
        print(f"示例{idx}：")
        print(f"  纯文件名：{pure_name}")
        print(f"  最终路径：{final_path}")
        print(f"  状态：{status}\n")

    # 关键提示：帮助用户定位问题
    print(f"=== 最终验证结果 ===")
    if exist_count == len(sample_imgs):
        print(f"✅ 所有抽样路径均有效！可直接用于训练～")
    else:
        print(f"❌ 部分/全部路径无效，请按以下步骤排查：")
        print(f"1. 确认 IMAGE_ABS_DIR 是图片实际存放目录（当前配置：{IMAGE_ABS_DIR}）")
        print(f"   - 执行命令 `ls {IMAGE_ABS_DIR}` 查看目录下是否有示例中的纯文件名（如 11 (1)_v1_heavy.jpg）")
        print(f"2. 若目录不存在/无图片：修正 IMAGE_ABS_DIR（比如图片在 /root/xxx/images/train/ 下，就更新配置）")
        print(f"3. 若文件名不一致：检查标注文件中的 file_name 与实际图片文件名是否完全匹配（区分大小写）")


if __name__ == "__main__":
    main()