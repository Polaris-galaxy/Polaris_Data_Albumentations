import os

def diagnose_directory_structure():
    """诊断目录结构"""
    print("=" * 60)
    print("目录结构诊断")
    print("=" * 60)
    
    # 检查当前目录
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    print(f"当前目录内容: {os.listdir('.')}")
    
    # 检查可能的目录结构
    possible_dirs = [
        "augmented_1000_fixed",
        "augmented_1000_final", 
        "original_data",
        "images",
        "labels"
    ]
    
    for dir_name in possible_dirs:
        if os.path.exists(dir_name):
            print(f"\n✅ 找到目录: {dir_name}")
            contents = os.listdir(dir_name)
            print(f"   内容: {contents}")
            
            # 检查子目录
            for item in contents:
                item_path = os.path.join(dir_name, item)
                if os.path.isdir(item_path):
                    sub_contents = os.listdir(item_path)
                    print(f"     📁 {item}/: {len(sub_contents)} 个项目")
                    # 显示前5个文件
                    if sub_contents:
                        print(f"         示例: {sub_contents[:5]}")
        else:
            print(f"❌ 目录不存在: {dir_name}")

def find_image_files():
    """查找所有图像文件"""
    print("\n" + "=" * 60)
    print("查找图像文件")
    print("=" * 60)
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []
    
    # 在当前目录及子目录中搜索图像文件
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower().endswith(image_extensions):
                relative_path = os.path.relpath(os.path.join(root, file))
                image_files.append(relative_path)
    
    print(f"找到 {len(image_files)} 个图像文件:")
    for img in image_files[:20]:  # 显示前20个
        print(f"  📷 {img}")
    
    if len(image_files) > 20:
        print(f"  ... 还有 {len(image_files) - 20} 个文件")
    
    return image_files

def find_label_files():
    """查找所有标签文件"""
    print("\n" + "=" * 60)
    print("查找标签文件")
    print("=" * 60)
    
    label_files = []
    
    # 在当前目录及子目录中搜索标签文件
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower().endswith('.txt'):
                relative_path = os.path.relpath(os.path.join(root, file))
                label_files.append(relative_path)
    
    print(f"找到 {len(label_files)} 个标签文件:")
    for label in label_files[:20]:  # 显示前20个
        print(f"  📄 {label}")
    
    if len(label_files) > 20:
        print(f"  ... 还有 {len(label_files) - 20} 个文件")
    
    return label_files

if __name__ == "__main__":
    diagnose_directory_structure()
    image_files = find_image_files()
    label_files = find_label_files()
    
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)
    print(f"总图像文件: {len(image_files)}")
    print(f"总标签文件: {len(label_files)}")
    
    if image_files and label_files:
        print("\n💡 建议的路径设置:")
        
        # 分析最常见的目录结构
        image_dirs = {}
        for img in image_files:
            dir_name = os.path.dirname(img)
            image_dirs[dir_name] = image_dirs.get(dir_name, 0) + 1
        
        label_dirs = {}
        for label in label_files:
            dir_name = os.path.dirname(label)
            label_dirs[dir_name] = label_dirs.get(dir_name, 0) + 1
        
        # 找到最可能的图像和标签目录
        most_common_image_dir = max(image_dirs.items(), key=lambda x: x[1])[0] if image_dirs else "."
        most_common_label_dir = max(label_dirs.items(), key=lambda x: x[1])[0] if label_dirs else "."
        
        print(f"IMAGE_DIR = '{most_common_image_dir}'")
        print(f"LABEL_DIR = '{most_common_label_dir}'")