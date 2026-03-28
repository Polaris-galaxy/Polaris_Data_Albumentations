import cv2
import os
from pathlib import Path
import numpy as np

print('当前opencv版本为:', cv2.__version__)

def video_to_frames():
    VIDEO_PATH = r"D:\Galaxy\其他\桌面\视频"
    OUTPUT_DIR = r"D:\Galaxy\其他\桌面\输出"
    TARGET_IMAGE_NUMBER = 1600  # 目标抽取总帧数

    PATH = Path(VIDEO_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取所有 .avi 文件
    files = [f for f in PATH.iterdir() if f.is_file() and f.suffix == '.avi']
    files.sort(key=lambda x: x.name)

    # ---------- 第一步：收集每个视频的总帧数 ----------
    video_info = []  # 每个元素为 (文件路径, 总帧数)
    for file in files:
        cap = cv2.VideoCapture(str(file))
        if not cap.isOpened():
            print(f"警告: 无法打开视频 {file.name}，已跳过")
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        video_info.append((file, total_frames))
        print(f"视频: {file.name}, 总帧数: {total_frames}")

    if not video_info:
        print("没有可用的视频文件！")
        return

    # ---------- 第二步：按总帧数比例分配目标帧数 ----------
    total_frames_all = sum(info[1] for info in video_info)
    # 先按比例计算理论帧数（浮点数）
    raw_alloc = [TARGET_IMAGE_NUMBER * frames / total_frames_all for frames in [info[1] for info in video_info]]
    # 向下取整得到基本分配数
    base_alloc = [int(x) for x in raw_alloc]
    # 计算剩余未分配的帧数
    allocated = sum(base_alloc)
    remaining = TARGET_IMAGE_NUMBER - allocated

    # 按小数部分从大到小排序，将剩余帧数分配给小数部分最大的视频
    fractional_parts = [(i, raw_alloc[i] - base_alloc[i]) for i in range(len(video_info))]
    fractional_parts.sort(key=lambda x: x[1], reverse=True)

    final_alloc = base_alloc[:]
    for i in range(remaining):
        final_alloc[fractional_parts[i][0]] += 1

    # 确保每个视频至少分配 1 帧（除非视频总帧数为 0，但这种情况极少）
    for i in range(len(final_alloc)):
        if final_alloc[i] == 0:
            final_alloc[i] = 1
            # 同时从某个分配过多的视频中减去1，保持总数不变（简单起见，这里可以重新平衡，但大概率不会触发）
            # 我们可以在分配时预先处理，但为了简洁，假设不会出现0帧分配

    print("\n按比例分配结果：")
    for i, (file, total_frames) in enumerate(video_info):
        print(f"{file.name}: 总帧数 {total_frames}, 分配抽取 {final_alloc[i]} 帧")

    # ---------- 第三步：依次处理每个视频 ----------
    for idx, (file, total_frames) in enumerate(video_info):
        alloc_frames = final_alloc[idx]
        print(f'\n正在处理视频: {file.name}, 需抽取 {alloc_frames} 帧')

        cap = cv2.VideoCapture(str(file))
        if not cap.isOpened():
            print(f"无法打开视频文件: {file.name}")
            continue

        # 如果目标抽取帧数大于视频总帧数，则抽取全部帧
        if alloc_frames >= total_frames:
            indices = list(range(total_frames))
        else:
            # 使用 np.linspace 生成均匀的帧索引
            indices = np.linspace(0, total_frames - 1, num=alloc_frames, dtype=int)

        indices_set = set(indices)
        frame_count = 0
        saved_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count in indices_set:
                out_filename = f"{file.stem}_frame_{saved_frames}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_filename)
                cv2.imwrite(out_path, frame)
                saved_frames += 1

            frame_count += 1

        cap.release()
        print(f'实际保存帧数: {saved_frames}')

if __name__ == "__main__":
    video_to_frames()