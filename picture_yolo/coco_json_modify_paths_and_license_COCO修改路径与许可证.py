import json
import argparse
import os

def modify_coco_json(input_json, output_json, image_path_prefix=None, fix_license=False, target_license_id=1):
    """
    修改 COCO 格式 JSON 文件。

    Args:
        input_json (str): 输入 JSON 文件的路径。
        output_json (str): 输出 JSON 文件的路径。
        image_path_prefix (str, optional): 新的图片路径前缀。如果为 None，则不修改图片路径。
                                           例如，原路径是 "train/image1.jpg"，前缀设为 "new_train"，
                                           新路径会变成 "new_train/image1.jpg"。
                                           如果前缀是绝对路径，如 "/data/dataset/train"，
                                           则会变成 "/data/dataset/train/image1.jpg"。
        fix_license (bool, optional): 是否修复 license ID 不匹配的问题。默认为 False。
        target_license_id (int, optional): 如果修复 license，目标的 license ID。默认为 1。
    """
    print(f"正在读取文件: {input_json}")
    # 1. 读取 JSON 文件
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 修改图片路径
    if image_path_prefix is not None:
        print(f"开始修改图片路径，新的路径前缀为: '{image_path_prefix}'")
        num_images = len(data.get('images', []))
        for i, image in enumerate(data['images']):
            # 原始文件名 (例如 "train/image1.jpg" 或 "image1.jpg")
            original_filename = image['file_name']
            
            # 获取文件名本身 (例如 "image1.jpg")
            base_filename = os.path.basename(original_filename)
            
            # 构建新的路径
            new_file_name = os.path.join(image_path_prefix, base_filename)
            
            # 在 Windows 上，os.path.join 会使用反斜杠 \，COCO 通常使用正斜杠 /
            # 为了兼容性，将路径分隔符统一替换为 /
            new_file_name = new_file_name.replace('\\', '/')
            
            image['file_name'] = new_file_name
            
            if (i + 1) % 100 == 0 or i + 1 == num_images:
                print(f"已处理 {i + 1}/{num_images} 张图片的路径。")
        print("图片路径修改完成。")
    else:
        print("未启用图片路径修改功能。")

    # 3. 修复 License ID 不匹配
    if fix_license:
        print(f"开始修复 License ID 不匹配问题，目标 License ID 为: {target_license_id}")
        
        # 确保 'licenses' 字段存在且不为空
        if 'licenses' not in data or not data['licenses']:
            print("警告: 'licenses' 字段为空，将创建一个默认的 license 条目。")
            data['licenses'] = [{
                "id": target_license_id,
                "name": "Unknown",
                "url": ""
            }]
        else:
            # 将所有 license 的 ID 都设置为目标 ID (通常只保留一个)
            # 为了严格匹配，我们保留第一个 license 条目并修改其 ID，删除其余的
            main_license = data['licenses'][0]
            main_license['id'] = target_license_id
            data['licenses'] = [main_license]
            print(f"已将 'licenses' 列表中的主要条目 ID 修改为 {target_license_id}。")

        # 将所有图片的 'license' 字段设置为目标 ID
        num_images = len(data.get('images', []))
        for i, image in enumerate(data['images']):
            image['license'] = target_license_id
            
            if (i + 1) % 100 == 0 or i + 1 == num_images:
                print(f"已处理 {i + 1}/{num_images} 张图片的 license 字段。")
        print("License ID 修复完成。")
    else:
        print("未启用 License ID 修复功能。")

    # 4. 保存修改后的 JSON 文件
    print(f"正在保存修改后的文件到: {output_json}")
    with open(output_json, 'w', encoding='utf-8') as f:
        # indent=4 保持 JSON 文件的可读性
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("所有操作完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="修改 COCO 格式 JSON 文件的图片路径和 License ID。")
    
    parser.add_argument("--input", type=str, required=True, help="输入的 COCO JSON 文件路径。")
    parser.add_argument("--output", type=str, required=True, help="修改后输出的 JSON 文件路径。")
    
    parser.add_argument("--image-prefix", type=str, default=None, 
                        help="新的图片路径前缀。例如 --image-prefix 'train2017' 会将路径从 'image1.jpg' 改为 'train2017/image1.jpg'。如果不提供，则不修改图片路径。")
    
    parser.add_argument("--fix-license", type=str, default="false", choices=["true", "false"],
                        help="是否修复 license ID 不匹配问题。'true' 表示修复，'false' 表示不修复。默认为 'false'。")
    
    parser.add_argument("--target-license-id", type=int, default=1,
                        help="当 --fix-license 为 'true' 时，目标的 license ID。默认为 1。")

    args = parser.parse_args()

    # 将 --fix-license 的字符串参数转换为布尔值
    fix_license_bool = args.fix_license.lower() == "true"

    modify_coco_json(
        input_json=args.input,
        output_json=args.output,
        image_path_prefix=args.image_prefix,
        fix_license=fix_license_bool,
        target_license_id=args.target_license_id
    )