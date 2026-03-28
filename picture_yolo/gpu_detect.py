import os
import sys
import cv2
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
from argparse import Namespace

# -------------------------- 导入官方 Deformable DETR 模块 --------------------------
try:
    from models.deformable_detr import build
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保此脚本位于 Deformable-DETR 仓库的根目录下。")
    sys.exit(1)

# -------------------------- 工具函数 --------------------------
def load_classes(classes_path):
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"类别文件不存在: {classes_path}")
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR 视频预测 (GPU 版)', add_help=False)
    parser.add_argument('--resume', type=str, required=True, help='训练权重路径')
    parser.add_argument('--video_path', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output_path', type=str, default='autodl-tmp/检测视频/output.mp4', help='输出视频路径')
    parser.add_argument('--classes_path', type=str, required=True, help='类别文件路径')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--input_size', type=int, nargs=2, default=[640, 480], help='输入尺寸 (宽, 高)')
    return parser

# -------------------------- 主函数 --------------------------
def main(args):
    classes = load_classes(args.classes_path)
    num_classes = len(classes)
    print(f"发现 {num_classes} 个类别: {classes}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"检测到 GPU: {torch.cuda.get_device_name(0)}，将使用 GPU 进行推理。")
    else:
        print("未检测到 GPU，将使用 CPU 进行推理（速度会非常慢）。")

    # ==================================================================
    # 完整参数配置（包含所有必需参数）
    # ==================================================================
    model_config = Namespace(
        # --- 模型结构核心参数 ---
        num_classes=num_classes,
        backbone='resnet50',
        dilation=False,
        position_embedding='sine',
        hidden_dim=256,
        num_queries=300,

        # Transformer 参数
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation='relu',
        pre_norm=False,

        # Deformable DETR 特定参数
        num_feature_levels=4,
        num_points=4,
        enc_n_points=4,
        dec_n_points=4,
        two_stage=False,
        with_box_refine=True,
        aux_loss=True,
        masks=False,

        # --- 损失函数与匹配参数 ---
        set_cost_class=1.0,
        set_cost_bbox=5.0,
        set_cost_giou=2.0,
        bbox_loss_coef=5.0,
        giou_loss_coef=2.0,
        cls_loss_coef=1.0,
        focal_alpha=0.25,
        eos_coef=0.1,

        # --- 设备和分布式训练参数 ---
        device=device,
        distributed=False,
        local_rank=-1,

        # --- 训练优化相关参数 ---
        lr=1e-4,
        lr_backbone=1e-5,
        weight_decay=1e-4,
        epochs=300,
        batch_size=2,
        clip_max_norm=0.1,

        # --- 其他杂项参数 ---
        dataset_file='coco',
        coco_path='dummy_path',
        output_dir='.',
        seed=42,
        resume='',
        start_epoch=0,
        print_freq=10,
        eval=False,
        frozen_weights=None,
        find_unused_parameters=False
    )

    # 4. 构建模型（关键修正：用 *rest 捕获多余返回值）
    print("正在构建模型...")
    try:
        model, *rest = build(model_config)  # 只提取第一个元素作为模型
    except (AttributeError, ValueError) as e:
        print(f"\n模型构建失败！")
        print(f"原始错误信息: {e}")
        sys.exit(1)
        
    model.to(device)  # 模型移到 GPU/CPU
    model.eval()      # 切换到推理模式
    print("模型构建成功！")

    # 5. 加载权重（兼容 DDP 训练权重）
    print(f"正在加载权重: {args.resume}")
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"权重文件不存在: {args.resume}")
    
    checkpoint = torch.load(args.resume, map_location=device)
    # 提取模型权重（处理不同训练框架的权重存储格式）
    state_dict = checkpoint.get('model', checkpoint)
    # 去除 DDP 训练添加的 "module." 前缀
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # 加载权重（strict=False 忽略无关参数，提高兼容性）
    model.load_state_dict(new_state_dict, strict=False)
    print("权重加载成功！")

    # 6. 视频读写初始化
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {args.video_path}")

    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\n--- 视频信息 ---")
    print(f"路径: {args.video_path}")
    print(f"分辨率: {frame_width}x{frame_height}")
    print(f"帧率 (FPS): {fps:.2f}")
    print(f"总帧数: {total_frames}")
    print(f"模型输入尺寸: {args.input_size[0]}x{args.input_size[1]}")
    print(f"置信度阈值: {args.confidence_threshold}")
    print("-----------------")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (frame_width, frame_height))

    # 7. 图像预处理（适配 Deformable DETR 输入要求）
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(args.input_size[::-1]),  # PIL.Resize 格式：(width, height)
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet 均值
            std=[0.229, 0.224, 0.225]    # ImageNet 标准差
        )
    ])

    # 8. 逐帧推理与结果绘制
    print("\n开始处理视频...")
    with torch.no_grad():  # 关闭梯度计算，加速推理并节省显存
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break  # 视频读取结束

            # 预处理：BGR -> RGB + 缩放 + 归一化 + 增加 batch 维度
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(frame_rgb).unsqueeze(0).to(device)  # shape: (1, 3, H, W)

            # 模型推理
            outputs = model(input_tensor)

            # 后处理：解析检测结果（筛选高置信度目标）
            pred_logits = outputs['pred_logits'][0].cpu()  # 预测类别得分：(num_queries, num_classes+1)
            pred_boxes = outputs['pred_boxes'][0].cpu()    # 预测框坐标（归一化）：(num_queries, 4)

            # 筛选高置信度目标（排除背景类）
            probas = pred_logits.softmax(-1)[:, :-1]  # 背景类是最后一列，排除后计算概率
            keep_mask = probas.max(-1).values > args.confidence_threshold  # 置信度筛选

            # 绘制检测框和类别标签
            if keep_mask.any():
                # 坐标反归一化：将 [0,1] 映射到视频实际分辨率
                bboxes_scaled = pred_boxes[keep_mask] * torch.tensor([
                    frame_width, frame_height, frame_width, frame_height
                ])
                class_indices = probas[keep_mask].argmax(-1)  # 预测类别索引
                confidences = probas[keep_mask].max(-1).values  # 预测置信度

                # 逐个绘制检测结果
                for (x1, y1, x2, y2), cls_idx, conf in zip(
                    bboxes_scaled.tolist(), class_indices.tolist(), confidences.tolist()
                ):
                    # 坐标转为整数（避免绘制异常）
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    # 绘制矩形框（绿色，线宽 2）
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 绘制类别+置信度标签（绿色，字体大小 0.6）
                    label_text = f"{classes[cls_idx]}: {conf:.2f}"
                    cv2.putText(
                        frame, label_text, (x1, max(0, y1 - 10)),  # 标签位置（框上方 10px）
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

            # 将绘制后的帧写入输出视频
            out.write(frame)

            # 打印处理进度（每 10 帧或最后一帧）
            if (frame_idx + 1) % 10 == 0 or (frame_idx + 1) == total_frames:
                progress = (frame_idx + 1) / total_frames * 100
                print(f"处理进度: {frame_idx + 1}/{total_frames} ({progress:.1f}%)")

    # 9. 释放所有资源
    cap.release()  # 释放视频读取器
    out.release()  # 释放视频写入器
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
    print(f"\n视频处理完成！输出文件已保存至: {args.output_path}")

# -------------------------- 入口函数 --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR 视频目标检测', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

    # 推荐使用绝对路径，避免所有路径问题
# python gpu_predict.py \
#   --resume "/root/Deformable-DETR/outputs/checkpoint0199.pth" \
#   --video_path "/root/autodl-tmp/检测视频/2025年10月26日 屏幕视频 00时06分38秒.mp4" \
#   --output_path "/root/autodl-tmp/检测视频/output.mp4" \
#   --classes_path "/root/Deformable-DETR/classes.txt" \
#   --confidence_threshold 0.5 \
#   --input_size 640 480