import os
import sys
import platform
import cv2
import torch
import argparse
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
    parser = argparse.ArgumentParser('Deformable DETR 视频/摄像头预测 (GPU 版)', add_help=False)
    parser.add_argument('--resume', type=str, required=True, help='训练权重路径')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['file', 'camera'],
        default='file',
        help='file=读取视频文件；camera=读取摄像头',
    )
    parser.add_argument(
        '--video_path',
        type=str,
        default=None,
        help='mode=file 时必填：输入视频路径',
    )
    parser.add_argument(
        '--camera_id',
        type=int,
        default=0,
        help='mode=camera 时摄像头设备号（默认 0）',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='autodl-tmp/检测视频/output.mp4',
        help='输出视频路径（与 --save_video / 文件模式配合）',
    )
    parser.add_argument('--classes_path', type=str, required=True, help='类别文件路径')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--input_size', type=int, nargs=2, default=[640, 480], help='输入尺寸 (宽, 高)')
    parser.add_argument(
        '--no_display',
        action='store_true',
        help='不弹窗（无显示器/SSH 时用；仅保存或仅跑推理）',
    )
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='不写入输出视频（仅实时显示或仅跑通推理）',
    )
    parser.add_argument(
        '--save_video',
        action='store_true',
        help='mode=camera 时默认不写文件；加上本项则同时录制到 --output_path',
    )
    return parser


def _open_video_capture(mode, video_path, camera_id):
    if mode == 'file':
        cap = cv2.VideoCapture(video_path)
        src_desc = video_path
    else:
        if platform.system() == 'Windows':
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_id)
        else:
            cap = cv2.VideoCapture(camera_id)
        src_desc = f"摄像头 device={camera_id}"
    return cap, src_desc


def _draw_detections(frame, pred_logits, pred_boxes, frame_width, frame_height,
                     classes, confidence_threshold):
    pred_logits = pred_logits.cpu()
    pred_boxes = pred_boxes.cpu()
    probas = pred_logits.softmax(-1)[:, :-1]
    keep_mask = probas.max(-1).values > confidence_threshold
    if not keep_mask.any():
        return frame
    scale = torch.tensor([frame_width, frame_height, frame_width, frame_height])
    bboxes_scaled = pred_boxes[keep_mask] * scale
    class_indices = probas[keep_mask].argmax(-1)
    confidences = probas[keep_mask].max(-1).values
    for (x1, y1, x2, y2), cls_idx, conf in zip(
        bboxes_scaled.tolist(), class_indices.tolist(), confidences.tolist()
    ):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{classes[cls_idx]}: {conf:.2f}"
        cv2.putText(
            frame, label_text, (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    return frame


# -------------------------- 主函数 --------------------------
def main(args):
    if args.mode == 'file':
        if not args.video_path:
            print("错误: mode=file 时必须提供 --video_path")
            sys.exit(1)
        if not os.path.isfile(args.video_path):
            raise FileNotFoundError(f"视频文件不存在: {args.video_path}")
    elif args.video_path:
        print("提示: mode=camera 将忽略 --video_path")

    classes = load_classes(args.classes_path)
    num_classes = len(classes)
    print(f"发现 {num_classes} 个类别: {classes}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"检测到 GPU: {torch.cuda.get_device_name(0)}，将使用 GPU 进行推理。")
    else:
        print("未检测到 GPU，将使用 CPU 进行推理（速度会非常慢）。")

    model_config = Namespace(
        num_classes=num_classes,
        backbone='resnet50',
        dilation=False,
        position_embedding='sine',
        hidden_dim=256,
        num_queries=300,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation='relu',
        pre_norm=False,
        num_feature_levels=4,
        num_points=4,
        enc_n_points=4,
        dec_n_points=4,
        two_stage=False,
        with_box_refine=True,
        aux_loss=True,
        masks=False,
        set_cost_class=1.0,
        set_cost_bbox=5.0,
        set_cost_giou=2.0,
        bbox_loss_coef=5.0,
        giou_loss_coef=2.0,
        cls_loss_coef=1.0,
        focal_alpha=0.25,
        eos_coef=0.1,
        device=device,
        distributed=False,
        local_rank=-1,
        lr=1e-4,
        lr_backbone=1e-5,
        weight_decay=1e-4,
        epochs=300,
        batch_size=2,
        clip_max_norm=0.1,
        dataset_file='coco',
        coco_path='dummy_path',
        output_dir='.',
        seed=42,
        resume='',
        start_epoch=0,
        print_freq=10,
        eval=False,
        frozen_weights=None,
        find_unused_parameters=False,
    )

    print("正在构建模型...")
    try:
        model, *rest = build(model_config)
    except (AttributeError, ValueError) as e:
        print(f"\n模型构建失败！\n原始错误信息: {e}")
        sys.exit(1)

    model.to(device)
    model.eval()
    print("模型构建成功！")

    print(f"正在加载权重: {args.resume}")
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"权重文件不存在: {args.resume}")

    checkpoint = torch.load(args.resume, map_location=device)
    state_dict = checkpoint.get('model', checkpoint)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    print("权重加载成功！")

    cap, src_desc = _open_video_capture(args.mode, args.video_path, args.camera_id)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频源: {src_desc}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1e-2:
        fps = 30.0 if args.mode == 'camera' else 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    want_save = not args.no_save
    if args.mode == 'camera' and not args.save_video:
        want_save = False
    if args.no_save:
        want_save = False

    out = None
    if want_save:
        out_dir = os.path.dirname(os.path.abspath(args.output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            print(f"警告: 无法创建 VideoWriter，将不保存视频: {args.output_path}")
            out = None
            want_save = False

    saved_any = want_save and out is not None

    print("\n--- 输入源 ---")
    print(f"模式: {args.mode}")
    print(f"源: {src_desc}")
    print(f"分辨率: {frame_width}x{frame_height}")
    if args.mode == 'file':
        print(f"帧率 (FPS): {fps:.2f}")
        print(f"总帧数: {total_frames if total_frames > 0 else '未知（将读到结束）'}")
    print(f"实时显示: {'关闭' if args.no_display else '开启（按 q 或 ESC 退出）'}")
    print(f"保存视频: {'是 -> ' + args.output_path if saved_any else '否'}")
    print(f"模型输入尺寸: {args.input_size[0]}x{args.input_size[1]}")
    print(f"置信度阈值: {args.confidence_threshold}")
    print("-----------------")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(args.input_size[::-1]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    window_name = "Deformable DETR"
    frame_idx = 0
    print("\n开始处理…")

    try:
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = transform(frame_rgb).unsqueeze(0).to(device)
                outputs = model(input_tensor)

                pred_logits = outputs['pred_logits'][0]
                pred_boxes = outputs['pred_boxes'][0]
                frame = _draw_detections(
                    frame, pred_logits, pred_boxes,
                    frame_width, frame_height,
                    classes, args.confidence_threshold,
                )

                if not args.no_display:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q'), 27):
                        print("用户按键退出。")
                        break

                if out is not None:
                    out.write(frame)

                frame_idx += 1
                if args.mode == 'file' and total_frames > 0:
                    if frame_idx % 10 == 0 or frame_idx == total_frames:
                        pct = frame_idx / total_frames * 100
                        print(f"处理进度: {frame_idx}/{total_frames} ({pct:.1f}%)")
                elif args.mode == 'camera' and frame_idx % 60 == 0:
                    print(f"已处理帧数: {frame_idx}")
    finally:
        cap.release()
        if out is not None:
            out.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if saved_any:
        print(f"\n视频已保存: {args.output_path}")
    print("结束。")


# -------------------------- 入口函数 --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR 视频/摄像头检测', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

# 示例（在 Deformable-DETR 仓库根目录执行）:
# 模式1 — 视频文件 + 实时窗口 + 保存
# python deformable_detr_gpu_infer_file_camera_GPU视频与摄像头检测.py --mode file --video_path "D:/data/in.mp4" --output_path "D:/data/out.mp4" \
#   --resume "outputs/checkpoint0199.pth" --classes_path "classes.txt"
#
# 模式2 — 摄像头 + 实时窗口（默认不录文件；要录请加 --save_video）
# python deformable_detr_gpu_infer_file_camera_GPU视频与摄像头检测.py --mode camera --camera_id 0 --save_video --output_path "D:/data/cam_out.mp4" \
#   --resume "outputs/checkpoint0199.pth" --classes_path "classes.txt"
#
# 仅显示、不保存：加 --no_save
# 无显示器：加 --no_display（可与保存组合）
