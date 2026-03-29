import os
import cv2
import numpy as np
import albumentations as A
import random
import shutil
import time
from typing import List, Tuple, Optional
from pathlib import Path


class FixedYOLOAugmentor:
    def __init__(self, image_dir: str, label_dir: str, output_dir: str, 
                 min_bbox_area: float = 0.00005,  # 最小边界框面积（相对整图），读取与过滤；小目标可调低
                 class_ids: List[int] = None,    # 已知有效类别ID（None则不校验）
                 use_vertical_flip: bool = False,  # 竖直翻转对方向敏感任务常产生不自然样本，默认关闭
                 bbox_min_visibility: float = 0.22,  # 几何增强后框保留比例下限；小目标宜略低于中大目标
                 require_prominent_labels: bool = True,  # True 时至少保留一个超过 prominent_min_* 的框（小目标需配合调低阈值）
                 prominent_min_area: float = 0.0001,   # 「参与输出」的相对面积下限，约 0.01% 整图（原为 0.25% 易筛掉小目标）
                 prominent_min_side: float = 0.006,  # 相对宽高中较短边下限，约千分之六（原 0.035 偏中大目标）
                 target_count: int = 1000,  # 期望生成的总张数（复制+各类增强合计）
                 # 小图：ROI 裁剪 → 放大 → 贴到更大画布（便于后续安全裁剪与几何增强）
                 enable_small_image_upscale_canvas: bool = True,
                 small_image_trigger_min_edge: int = 224,  # min(H,W) 小于此则尝试预处理
                 small_image_trigger_max_area: Optional[int] = 180_000,  # 像素 H*W 小于此也会触发；None 表示不按面积触发
                 small_image_roi_margin_frac: float = 0.12,  # 相对「原图短边」的外扩边距
                 small_image_upscale_min_short_edge: int = 448,  # ROI 放大后短边目标下限
                 small_image_canvas_margin_frac: float = 0.18):  # 画布相对 ROI 的额外留白比例（四边合计）
        self.image_dir = Path(image_dir).resolve()
        self.label_dir = Path(label_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.target_count = max(1, int(target_count))
        self.min_bbox_area = min_bbox_area  # 过滤过小边界框（相对面积）
        self.class_ids = set(class_ids) if class_ids is not None else None
        self.use_vertical_flip = use_vertical_flip
        self.bbox_min_visibility = bbox_min_visibility
        self.require_prominent_labels = require_prominent_labels
        self.prominent_min_area = prominent_min_area
        self.prominent_min_side = prominent_min_side
        self.enable_small_image_upscale_canvas = enable_small_image_upscale_canvas
        self.small_image_trigger_min_edge = int(small_image_trigger_min_edge)
        self.small_image_trigger_max_area = small_image_trigger_max_area
        self.small_image_roi_margin_frac = float(small_image_roi_margin_frac)
        self.small_image_upscale_min_short_edge = int(small_image_upscale_min_short_edge)
        self.small_image_canvas_margin_frac = float(small_image_canvas_margin_frac)
        
        # 创建输出目录
        self.output_img_dir = self.output_dir / 'images'
        self.output_label_dir = self.output_dir / 'labels'
        self.output_img_dir.mkdir(parents=True, exist_ok=True)
        self.output_label_dir.mkdir(parents=True, exist_ok=True)
        
        # 记录处理过的文件名，避免重复
        self.processed_names = set()

    def _get_matching_label(self, image_path: Path) -> Path:
        """获取与图像严格匹配的标注文件（处理多种扩展名和大小写）"""
        # 尝试常见标注文件格式：同文件名+.txt（忽略图像扩展名）
        label_stem = image_path.stem
        possible_labels = [
            self.label_dir / f"{label_stem}.txt",
            self.label_dir / f"{label_stem.lower()}.txt",
            self.label_dir / f"{label_stem.upper()}.txt"
        ]
        for lbl in possible_labels:
            if lbl.exists():
                return lbl
        return None

    def read_yolo_annotations(self, label_path: Path, img_width: int, img_height: int) -> List[Tuple]:
        """读取YOLO格式标注并进行严格验证"""
        annotations = []
        if not label_path.exists():
            return annotations
            
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    print(f"警告: {label_path} 第{line_num}行格式错误（需5个字段），跳过")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center_rel = float(parts[1])
                    y_center_rel = float(parts[2])
                    width_rel = float(parts[3])
                    height_rel = float(parts[4])
                    
                    # 1. 验证坐标范围（严格0-1）
                    if not (0 <= x_center_rel <= 1 and 0 <= y_center_rel <= 1):
                        print(f"警告: {label_path} 第{line_num}行中心坐标超出范围，跳过")
                        continue
                    if not (0 < width_rel <= 1 and 0 < height_rel <= 1):
                        print(f"警告: {label_path} 第{line_num}行宽高超出范围，跳过")
                        continue
                    
                    # 2. 验证边界框面积（避免过小框）
                    bbox_area = width_rel * height_rel
                    if bbox_area < self.min_bbox_area:
                        print(f"警告: {label_path} 第{line_num}行边界框过小（面积{bbox_area:.6f}），跳过")
                        continue
                    
                    # 3. 验证类别ID（如果已知有效类别）
                    if self.class_ids is not None and class_id not in self.class_ids:
                        print(f"警告: {label_path} 第{line_num}行类别ID {class_id} 无效，跳过")
                        continue
                    
                    annotations.append((class_id, x_center_rel, y_center_rel, width_rel, height_rel))
                    
                except ValueError as e:
                    print(f"警告: {label_path} 第{line_num}行数值解析错误: {e}，跳过")
                    continue
                    
        except Exception as e:
            print(f"错误: 读取标注 {label_path} 失败: {e}")
            
        return annotations

    def _bbox_is_prominent(self, ann: Tuple) -> bool:
        """相对坐标下判断框是否足够大、便于肉眼与检测头学习（面积 + 最短边）。"""
        _c, _x, _y, bw, bh = ann
        if bw * bh < self.prominent_min_area:
            return False
        if min(bw, bh) < self.prominent_min_side:
            return False
        return True

    def _filter_prominent(self, anns: List[Tuple]) -> List[Tuple]:
        if not self.require_prominent_labels:
            return list(anns)
        return [a for a in anns if self._bbox_is_prominent(a)]

    def convert_to_yolo_format(self, bboxes: List[Tuple]) -> List[str]:
        """转换为YOLO格式并进行最终校验"""
        yolo_lines = []
        for bbox in bboxes:
            class_id, x_center, y_center, width, height = bbox
            
            # 强制裁剪到有效范围（避免增强后微小越界）
            x_center = max(0.001, min(0.999, x_center))
            y_center = max(0.001, min(0.999, y_center))
            width = max(0.001, min(0.999 - x_center*2, width))  # 确保不超出边界
            height = max(0.001, min(0.999 - y_center*2, height))
            
            # 再次验证面积
            if width * height < self.min_bbox_area:
                print(f"过滤增强后过小边界框（面积{width*height:.6f}）")
                continue
            if self.require_prominent_labels and not self._bbox_is_prominent(
                (class_id, x_center, y_center, width, height)
            ):
                continue
            
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
        return yolo_lines

    def get_augmentation(self, aug_type: str):
        """获取增强变换（偏真实成像：压缩、噪点、轻透视；弱化易产生伪影的 CLAHE）"""
        bbox_params = self._bbox_params()

        vflip = [A.VerticalFlip(p=0.25)] if self.use_vertical_flip else []

        # 各强度共用的「相机/链路」扰动（不改变几何， bbox 安全）
        def camera_artifacts(quality_lo: int, quality_hi: int, noise_p: float):
            return [
                A.ImageCompression(quality_lower=quality_lo, quality_upper=quality_hi, p=0.35),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.15), p=noise_p),
                A.GaussNoise(std_range=(0.02, 0.08), mean_range=(0.0, 0.0), p=noise_p * 0.6),
            ]

        if aug_type == "light":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                *vflip,
                A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.45),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=8, p=0.25),
                *camera_artifacts(75, 98, 0.12),
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                    A.Sharpen(alpha=(0.1, 0.25), lightness=(0.5, 1.0), p=1.0),
                ], p=0.18),
            ], bbox_params=bbox_params)

        if aug_type == "medium":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                *vflip,
                A.ShiftScaleRotate(
                    shift_limit=0.04, scale_limit=0.12, rotate_limit=12,
                    border_mode=cv2.BORDER_REFLECT_101, p=0.55,
                ),
                A.Perspective(scale=(0.02, 0.045), p=0.22),
                A.RandomBrightnessContrast(brightness_limit=0.22, contrast_limit=0.22, p=0.5),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=28, val_shift_limit=15, p=0.38),
                *camera_artifacts(65, 92, 0.2),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.22),
                A.RandomGamma(gamma_limit=(0.85, 1.15), p=0.2),
            ], bbox_params=bbox_params)

        # heavy：更强但仍避免过高比例「油画感」本地对比度增强
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            *vflip,
            A.ShiftScaleRotate(
                shift_limit=0.06, scale_limit=0.18, rotate_limit=18,
                border_mode=cv2.BORDER_REFLECT_101, p=0.6,
            ),
            A.Perspective(scale=(0.03, 0.06), p=0.28),
            A.RandomBrightnessContrast(brightness_limit=0.32, contrast_limit=0.32, p=0.55),
            A.HueSaturationValue(hue_shift_limit=22, sat_shift_limit=35, val_shift_limit=22, p=0.45),
            *camera_artifacts(55, 88, 0.28),
            A.RandomFog(fog_coef_lower=0.08, fog_coef_upper=0.22, alpha_coef=0.1, p=0.15),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                A.Defocus(radius=(2, 4), p=1.0),
            ], p=0.3),
            A.RandomGamma(gamma_limit=(0.78, 1.22), p=0.28),
            A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=0.08),
        ], bbox_params=bbox_params)

    def _bbox_params(self) -> A.BboxParams:
        return A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=self.bbox_min_visibility,
            min_area=self.min_bbox_area,
            clip=True,
        )

    @staticmethod
    def _yolo_to_crop_normalized(
        xc: float, yc: float, bw: float, bh: float,
        W: int, H: int, x0: int, y0: int, cw: int, ch: int,
    ) -> Optional[Tuple[float, float, float, float]]:
        """将原图 YOLO 框转为裁剪片内的相对坐标；与裁剪区域求交，超出部分裁掉。"""
        w_abs = bw * W
        h_abs = bh * H
        cx = xc * W
        cy = yc * H
        bx1 = cx - w_abs / 2
        by1 = cy - h_abs / 2
        bx2 = cx + w_abs / 2
        by2 = cy + h_abs / 2
        ix1 = max(bx1, float(x0))
        iy1 = max(by1, float(y0))
        ix2 = min(bx2, float(x0 + cw))
        iy2 = min(by2, float(y0 + ch))
        if ix2 - ix1 < 1.0 or iy2 - iy1 < 1.0:
            return None
        icx = (ix1 + ix2) / 2
        icy = (iy1 + iy2) / 2
        iw = ix2 - ix1
        ih = iy2 - iy1
        return ((icx - x0) / cw, (icy - y0) / ch, iw / cw, ih / ch)

    def _union_boxes_pixels(
        self, bboxes: List[List[float]], W: int, H: int,
    ) -> Tuple[float, float, float, float]:
        x1m, y1m = float(W), float(H)
        x2m, y2m = 0.0, 0.0
        for xc, yc, bw, bh in bboxes:
            w_abs = bw * W
            h_abs = bh * H
            cx = xc * W
            cy = yc * H
            x1m = min(x1m, cx - w_abs / 2)
            y1m = min(y1m, cy - h_abs / 2)
            x2m = max(x2m, cx + w_abs / 2)
            y2m = max(y2m, cy + h_abs / 2)
        return x1m, y1m, x2m, y2m

    def _prepare_small_image_crop_upscale_canvas(
        self,
        image_rgb: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int],
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        对面积/分辨率偏小的图：按标注并集外扩裁 ROI，双三次放大到目标短边，再随机贴到更大画布上，
        相当于「裁一块 → 放大 → 拼进带边的大图」，后续 Albumentations 更有操作空间。
        若非触发条件或失败则原样返回。
        """
        if not self.enable_small_image_upscale_canvas or not bboxes:
            return image_rgb, bboxes, class_labels
        H, W = image_rgb.shape[:2]
        short = min(H, W)
        area = H * W
        by_edge = short < self.small_image_trigger_min_edge
        by_area = (
            self.small_image_trigger_max_area is not None
            and area < self.small_image_trigger_max_area
        )
        if not (by_edge or by_area):
            return image_rgb, bboxes, class_labels

        x1m, y1m, x2m, y2m = self._union_boxes_pixels(bboxes, W, H)
        margin = self.small_image_roi_margin_frac * float(short)
        x0 = max(0, int(np.floor(x1m - margin)))
        y0 = max(0, int(np.floor(y1m - margin)))
        x1c = min(W, int(np.ceil(x2m + margin)))
        y1c = min(H, int(np.ceil(y2m + margin)))
        cw = x1c - x0
        ch = y1c - y0
        if cw < 2 or ch < 2:
            return image_rgb, bboxes, class_labels

        crop = image_rgb[y0:y1c, x0:x1c]
        new_boxes: List[List[float]] = []
        new_classes: List[int] = []
        for lab, (xc, yc, bw, bh) in zip(class_labels, bboxes):
            t = self._yolo_to_crop_normalized(xc, yc, bw, bh, W, H, x0, y0, cw, ch)
            if t is None:
                continue
            nxc, nyc, nbw, nbh = t
            if nbw * nbh < self.min_bbox_area:
                continue
            new_boxes.append([nxc, nyc, nbw, nbh])
            new_classes.append(lab)
        if not new_boxes:
            return image_rgb, bboxes, class_labels

        ph, pw = crop.shape[:2]
        tgt = self.small_image_upscale_min_short_edge
        scale = max(1.0, float(tgt) / float(min(ph, pw)))
        new_w = max(1, int(round(pw * scale)))
        new_h = max(1, int(round(ph * scale)))
        if scale > 1.001:
            patch = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            patch = crop

        mf = max(0.0, self.small_image_canvas_margin_frac)
        Cw = max(new_w + 1, int(np.ceil(new_w * (1.0 + 2.0 * mf))))
        Ch = max(new_h + 1, int(np.ceil(new_h * (1.0 + 2.0 * mf))))
        border = np.concatenate(
            [patch[0, :, :], patch[-1, :, :], patch[:, 0, :], patch[:, -1, :]], axis=0
        )
        fill = tuple(int(x) for x in border.mean(axis=0))
        canvas = np.empty((Ch, Cw, 3), dtype=np.uint8)
        canvas[:, :] = fill
        ox = random.randint(0, max(0, Cw - new_w))
        oy = random.randint(0, max(0, Ch - new_h))
        canvas[oy:oy + new_h, ox:ox + new_w] = patch

        out_boxes: List[List[float]] = []
        out_classes: List[int] = []
        for lab, (xc, yc, bw, bh) in zip(new_classes, new_boxes):
            nxc = (xc * new_w + ox) / Cw
            nyc = (yc * new_h + oy) / Ch
            nbw = bw * new_w / Cw
            nbh = bh * new_h / Ch
            if not (0.001 <= nxc <= 0.999 and 0.001 <= nyc <= 0.999):
                continue
            if nbw * nbh < self.min_bbox_area:
                continue
            out_boxes.append([nxc, nyc, nbw, nbh])
            out_classes.append(lab)
        if not out_boxes:
            return image_rgb, bboxes, class_labels
        return canvas, out_boxes, out_classes

    def _apply_bbox_safe_crop(
        self,
        image_rgb: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int],
        aug_type: str,
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """在单图增强前随机做一次「含框安全」裁剪，放大目标、模拟近景/二次构图。"""
        h, w = image_rgb.shape[:2]
        if h < 64 or w < 64:
            return image_rgb, bboxes, class_labels
        p = {'light': 0.18, 'medium': 0.4, 'heavy': 0.52}.get(aug_type, 0.35)
        if random.random() > p or not bboxes:
            return image_rgb, bboxes, class_labels
        min_h = max(48, int(h * 0.52))
        min_w = max(48, int(w * 0.52))
        try:
            safe_crop = A.RandomSizedBBoxSafeCrop(
                min_max_height=(min_h, h),
                min_max_width=(min_w, w),
                w2h_ratio=(0.88, 1.12),
                erosion_rate=0.02,
                p=1.0,
            )
        except TypeError:
            # 旧版 albumentations 无 erosion_rate
            safe_crop = A.RandomSizedBBoxSafeCrop(
                min_max_height=(min_h, h),
                min_max_width=(min_w, w),
                w2h_ratio=(0.88, 1.12),
                p=1.0,
            )
        crop_compose = A.Compose([safe_crop], bbox_params=self._bbox_params())
        try:
            out = crop_compose(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
            if not out['bboxes']:
                return image_rgb, bboxes, class_labels
            return out['image'], out['bboxes'], out['class_labels']
        except Exception:
            return image_rgb, bboxes, class_labels

    def _create_pair_concat(
        self,
        image_paths: List[Path],
        label_paths: List[Path],
        output_name: str,
        axis: str,
    ) -> bool:
        """双图拼接：horizontal → 左右并排；vertical → 上下堆叠（标注同步变换）。"""
        if len(image_paths) != 2 or len(label_paths) != 2:
            return False
        axis = axis.lower()
        if axis not in ('horizontal', 'vertical'):
            return False
        try:
            tiles = []
            anns_per_tile = []
            target_wh: Optional[Tuple[int, int]] = None

            for img_path, label_path in zip(image_paths, label_paths):
                image = cv2.imread(str(img_path))
                if image is None:
                    return False
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tw, th = rgb.shape[1], rgb.shape[0]
                if target_wh is None:
                    target_wh = (tw, th)
                else:
                    if (tw, th) != target_wh:
                        rgb = cv2.resize(rgb, target_wh)
                        tw, th = target_wh
                tiles.append(rgb)
                annotations = self.read_yolo_annotations(label_path, tw, th)
                if not annotations:
                    return False
                anns_per_tile.append(annotations)

            assert target_wh is not None
            tw, th = target_wh

            if axis == 'horizontal':
                out_w, out_h = tw * 2, th
                offsets = [(0, 0), (tw, 0)]
            else:
                out_w, out_h = tw, th * 2
                offsets = [(0, 0), (0, th)]

            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            merged: List[Tuple] = []
            for tile, anns, (ox, oy) in zip(tiles, anns_per_tile, offsets):
                h_tile, w_tile = tile.shape[:2]
                canvas[oy:oy + h_tile, ox:ox + w_tile] = tile
                for class_id, xc, yc, bw, bh in anns:
                    xa = xc * w_tile + ox
                    ya = yc * h_tile + oy
                    bw_abs = bw * w_tile
                    bh_abs = bh * h_tile
                    merged.append((
                        class_id,
                        xa / out_w,
                        ya / out_h,
                        bw_abs / out_w,
                        bh_abs / out_h,
                    ))

            if not merged:
                return False
            merged = self._filter_prominent(merged)
            if not merged:
                return False
            output_name = self._check_duplicate_name(output_name)
            yolo_lines = self.convert_to_yolo_format(merged)
            out_img = self.output_img_dir / f"{output_name}.jpg"
            if not cv2.imwrite(str(out_img), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
                return False
            out_lbl = self.output_label_dir / f"{output_name}.txt"
            with open(out_lbl, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
            return True
        except Exception as e:
            _axis_cn = "横向" if axis == "horizontal" else "纵向"
            print(f"双图拼接错误（{_axis_cn}）: {e}，跳过")
            return False

    def _check_duplicate_name(self, output_name: str) -> str:
        """确保输出文件名唯一（避免重复）"""
        original_name = output_name
        suffix = 1
        while output_name in self.processed_names:
            output_name = f"{original_name}_{suffix}"
            suffix += 1
        self.processed_names.add(output_name)
        return output_name

    def augment_image(self, image_path: Path, label_path: Path, aug_type: str, base_name: str) -> bool:
        """增强单张图像（强化错误处理和有效性验证）"""
        try:
            # 1. 验证图像有效性
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"错误: 无法读取图像 {image_path}，跳过")
                return False
            if image.shape[0] < 32 or image.shape[1] < 32:  # 过滤过小图像
                print(f"错误: 图像 {image_path} 尺寸过小（小于 32×32 像素），跳过")
                return False
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            
            # 2. 读取并验证标注
            annotations = self.read_yolo_annotations(label_path, width, height)
            if not annotations:
                print(f"警告: 图像 {image_path} 无有效标注，跳过增强")
                return False
            work_anns = self._filter_prominent(annotations)
            if self.require_prominent_labels and not work_anns:
                print(f"警告: 图像 {image_path} 无「明显」标注（可调明显框面积/边长下限等参数），跳过增强")
                return False
            
            # 3. 准备增强数据（仅对已满足明显阈值的框做变换，避免小框拖垮增强）
            bboxes = []
            class_labels = []
            for ann in work_anns:
                class_id, x_center_rel, y_center_rel, width_rel, height_rel = ann
                bboxes.append([x_center_rel, y_center_rel, width_rel, height_rel])
                class_labels.append(class_id)
            
            # 4. 小图：ROI 放大并贴入大画布，再进入后续裁剪/增强（可与下步 RandomSizedBBoxSafeCrop 协同）
            image_rgb, bboxes, class_labels = self._prepare_small_image_crop_upscale_canvas(
                image_rgb, bboxes, class_labels
            )
            # 5. BBox 安全裁剪（可选）+ 应用其余增强
            image_rgb, bboxes, class_labels = self._apply_bbox_safe_crop(
                image_rgb, bboxes, class_labels, aug_type
            )
            transform = self.get_augmentation(aug_type)
            try:
                transformed = transform(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
            except Exception as e:
                print(f"增强变换失败 {image_path}: {e}，跳过")
                return False
            
            # 6. 验证增强后边界框
            transformed_bboxes = []
            for bbox, class_id in zip(transformed['bboxes'], transformed['class_labels']):
                x_center, y_center, bbox_width, bbox_height = bbox
                # 严格验证范围
                if not (0.001 <= x_center <= 0.999 and 0.001 <= y_center <= 0.999):
                    continue
                if not (0.001 <= bbox_width <= 0.999 and 0.001 <= bbox_height <= 0.999):
                    continue
                # 验证面积
                if bbox_width * bbox_height < self.min_bbox_area:
                    continue
                transformed_bboxes.append((class_id, x_center, y_center, bbox_width, bbox_height))
            
            if not transformed_bboxes:
                print(f"增强后无有效边界框 {image_path}，跳过")
                return False
            if self.require_prominent_labels:
                transformed_bboxes = self._filter_prominent(transformed_bboxes)
            if not transformed_bboxes:
                print(f"增强后无「明显」边界框 {image_path}，跳过")
                return False
            
            # 7. 生成唯一文件名并保存
            output_name = self._check_duplicate_name(f"{base_name}_{aug_type}")
            yolo_lines = self.convert_to_yolo_format(transformed_bboxes)
            
            # 保存图像
            output_image_path = self.output_img_dir / f"{output_name}.jpg"
            if not cv2.imwrite(str(output_image_path), cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)):
                print(f"错误: 无法保存增强图像 {output_image_path}，跳过")
                return False
            
            # 保存标注
            output_label_path = self.output_label_dir / f"{output_name}.txt"
            with open(output_label_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
            
            return True
            
        except Exception as e:
            print(f"增强图像 {image_path} 出错: {e}，跳过")
            return False

    def create_mosaic(self, image_paths: List[Path], label_paths: List[Path], output_name: str) -> bool:
        """创建马赛克增强（统一图像尺寸，强化校验）"""
        try:
            if len(image_paths) != 4:
                return False
            
            images = []
            all_annotations = []
            target_size = None  # 统一马赛克子图尺寸
            
            # 1. 读取并预处理4张图像（统一尺寸）
            for img_path, label_path in zip(image_paths, label_paths):
                # 读取图像
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"马赛克错误: 无法读取 {img_path}，跳过")
                    return False
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image_rgb.shape[:2]
                
                # 统一尺寸（以第一张图像为基准）
                if target_size is None:
                    target_size = (w, h)  # (width, height)
                else:
                    if (w, h) != target_size:
                        # 缩放至目标尺寸（避免马赛克拼接变形）
                        image_rgb = cv2.resize(image_rgb, target_size)
                        h, w = target_size[1], target_size[0]  # resize后h=target_height, w=target_width
                
                images.append(image_rgb)
                
                # 读取标注
                annotations = self.read_yolo_annotations(label_path, w, h)
                if not annotations:
                    print(f"马赛克警告: {img_path} 无有效标注，跳过此组合")
                    return False
                all_annotations.append(annotations)
            
            # 2. 创建马赛克画布（2x2拼接）
            mosaic_width = 2 * target_size[0]
            mosaic_height = 2 * target_size[1]
            mosaic_image = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
            
            # 3. 拼接图像并转换标注
            positions = [
                (0, 0),  # 左上
                (target_size[0], 0),  # 右上
                (0, target_size[1]),  # 左下  
                (target_size[0], target_size[1])  # 右下
            ]
            
            mosaic_annotations = []
            for img, annotations, (x, y) in zip(images, all_annotations, positions):
                h, w = img.shape[:2]
                mosaic_image[y:y+h, x:x+w] = img  # 拼接图像
                
                # 转换标注坐标
                for ann in annotations:
                    class_id, x_center_rel, y_center_rel, width_rel, height_rel = ann
                    
                    # 计算马赛克中的绝对坐标
                    x_center_abs = x_center_rel * w + x
                    y_center_abs = y_center_rel * h + y
                    width_abs = width_rel * w
                    height_abs = height_rel * h
                    
                    # 转换为相对马赛克的坐标
                    x_center_new = x_center_abs / mosaic_width
                    y_center_new = y_center_abs / mosaic_height
                    width_new = width_abs / mosaic_width
                    height_new = height_abs / mosaic_height
                    
                    # 验证有效性
                    if (0.001 <= x_center_new <= 0.999 and 
                        0.001 <= y_center_new <= 0.999 and 
                        0.001 <= width_new <= 0.999 and 
                        0.001 <= height_new <= 0.999 and 
                        width_new * height_new >= self.min_bbox_area):
                        mosaic_annotations.append((class_id, x_center_new, y_center_new, width_new, height_new))
            
            if not mosaic_annotations:
                print(f"马赛克无有效标注 {output_name}，跳过")
                return False
            mosaic_annotations = self._filter_prominent(mosaic_annotations)
            if not mosaic_annotations:
                print(f"马赛克无「明显」标注 {output_name}，跳过")
                return False
            
            # 4. 保存马赛克结果
            output_name = self._check_duplicate_name(output_name)
            yolo_lines = self.convert_to_yolo_format(mosaic_annotations)
            
            output_image_path = self.output_img_dir / f"{output_name}.jpg"
            if not cv2.imwrite(str(output_image_path), cv2.cvtColor(mosaic_image, cv2.COLOR_RGB2BGR)):
                print(f"错误: 无法保存马赛克图像 {output_image_path}，跳过")
                return False
            
            output_label_path = self.output_label_dir / f"{output_name}.txt"
            with open(output_label_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
            
            return True
            
        except Exception as e:
            print(f"马赛克创建错误: {e}，跳过")
            return False

    def augment_to_1000(self):
        """增强到 self.target_count 张（优化流程和校验）；目标张数在构造 FixedYOLOAugmentor 时传入 target_count。"""
        # 1. 获取并校验图像-标注对（严格匹配）
        valid_pairs = []  # 存储(图像路径, 标注路径)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for img_path in self.image_dir.glob('*.*'):
            if img_path.suffix.lower() not in image_extensions:
                continue
            # 严格匹配标注文件
            label_path = self._get_matching_label(img_path)
            if label_path is None:
                print(f"警告: 图像 {img_path.name} 无匹配标注文件，跳过")
                continue
            valid_pairs.append((img_path, label_path))
        
        original_count = len(valid_pairs)
        print(f"有效图像-标注对数量: {original_count}")
        
        if original_count == 0:
            print("❌ 没有找到有效图像-标注对")
            return

        if self.require_prominent_labels:
            print(
                f"📌 明显框约定（相对整图）：面积 ≥ {self.prominent_min_area}，"
                f"且 min(宽,高) ≥ {self.prominent_min_side}"
            )
            refined = []
            for img_path, label_path in valid_pairs:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                ih, iw = image.shape[:2]
                anns = self.read_yolo_annotations(label_path, iw, ih)
                if self._filter_prominent(anns):
                    refined.append((img_path, label_path))
            print(f"   其中至少含一个明显框、会参与输出: {len(refined)} / {original_count}")
            valid_pairs = refined
            if not valid_pairs:
                print("❌ 没有满足明显框阈值的样本，请将明显框面积下限或边长下限调低，或检查标注")
                return
        
        # 2. 复制原始数据（带校验）；若要求明显框，则只写入通过阈值的标注行
        print("📁 复制原始图片和标注...")
        copied_count = 0
        for img_path, label_path in valid_pairs:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"警告: 无法读取 {img_path}，跳过复制")
                continue
            ih, iw = image.shape[:2]
            anns = self.read_yolo_annotations(label_path, iw, ih)
            prominent_anns = self._filter_prominent(anns)
            if self.require_prominent_labels and not prominent_anns:
                continue
            # 生成唯一文件名
            base_name = f"original_{img_path.stem}"
            output_name = self._check_duplicate_name(base_name)
            
            # 复制图像
            img_dst = self.output_img_dir / f"{output_name}{img_path.suffix}"
            if not img_dst.exists():
                try:
                    shutil.copy2(img_path, img_dst)
                except Exception as e:
                    print(f"复制图像 {img_path} 失败: {e}，跳过")
                    continue
            
            # 写入过滤后的标注
            label_dst = self.output_label_dir / f"{output_name}.txt"
            if not label_dst.exists():
                try:
                    lines_out = self.convert_to_yolo_format(prominent_anns if self.require_prominent_labels else anns)
                    if not lines_out:
                        if img_dst.exists():
                            os.remove(img_dst)
                        continue
                    with open(label_dst, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines_out))
                except Exception as e:
                    print(f"写入标注 {label_dst} 失败: {e}，删除对应图像")
                    if img_dst.exists():
                        os.remove(img_dst)
                    continue
            
            copied_count += 1
        
        print(f"✅ 复制原始数据: {copied_count} 对")
        if copied_count == 0:
            print("❌ 无法复制原始数据，终止增强")
            return
        
        # 3. 基础增强（每张生成多个版本）
        print("🔄 开始基础增强...")
        base_count = 0
        augmentation_types = ['light', 'medium', 'heavy']
        versions_per_type = 2  # 每种增强类型生成的版本数
        
        for i, (img_path, label_path) in enumerate(valid_pairs):
            base_name = img_path.stem
            # 每种增强类型生成多个版本
            for aug_type in augmentation_types:
                for ver in range(versions_per_type):
                    success = self.augment_image(img_path, label_path, aug_type, f"{base_name}_v{ver+1}")
                    if success:
                        base_count += 1
            
            # 进度提示
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i+1}/{original_count} 张，生成基础增强 {base_count} 张")
        
        print(f"✅ 基础增强完成: {base_count} 张")
        
        # 4. 马赛克增强（选择有效图像组合）
        print("🧩 开始马赛克增强...")
        mosaic_count = 0
        # 计算需要的马赛克数量（避免过度生成）
        current_total = copied_count + base_count
        remaining = self.target_count - current_total
        mosaic_cap = min(200, max(50, self.target_count // 5))
        target_mosaic = min(mosaic_cap, max(0, remaining // 2))
        
        for i in range(target_mosaic):
            if len(valid_pairs) < 4:
                break
            # 随机选择4个不同的图像-标注对
            selected = random.sample(valid_pairs, 4)
            selected_imgs = [p[0] for p in selected]
            selected_labels = [p[1] for p in selected]
            
            success = self.create_mosaic(selected_imgs, selected_labels, f"mosaic_{i+1:04d}")
            if success:
                mosaic_count += 1
            
            if (i + 1) % 20 == 0:
                print(f"  已生成马赛克 {i+1}/{target_mosaic} 张")
        
        print(f"✅ 马赛克增强完成: {mosaic_count} 张")
        
        # 4b. 双图拼接（横向 / 纵向），补充多尺度构图
        print("⮂ 双图拼接（横/纵）...")
        current_total = copied_count + base_count + mosaic_count
        remaining_to_goal = self.target_count - current_total
        pair_cap = min(120, max(40, self.target_count // 8))
        target_pair = min(pair_cap, max(0, remaining_to_goal // 3))
        pair_h_count = 0
        pair_v_count = 0
        for i in range(target_pair):
            if len(valid_pairs) < 2:
                break
            a, b = random.sample(valid_pairs, 2)
            axis = 'horizontal' if random.random() < 0.55 else 'vertical'
            prefix = 'pair_h' if axis == 'horizontal' else 'pair_v'
            ok = self._create_pair_concat(
                [a[0], b[0]], [a[1], b[1]], f"{prefix}_{i+1:04d}", axis
            )
            if ok:
                if axis == 'horizontal':
                    pair_h_count += 1
                else:
                    pair_v_count += 1
            if (i + 1) % 30 == 0:
                print(f"  已尝试双图拼接 {i+1}/{target_pair}（成功 横向:{pair_h_count} 纵向:{pair_v_count}）")
        pair_total = pair_h_count + pair_v_count
        print(f"✅ 双图拼接完成: 横向 {pair_h_count} 张，纵向 {pair_v_count} 张")
        
        # 5. 额外增强（补充至 target_count 张）
        current_total = copied_count + base_count + mosaic_count + pair_total
        remaining = self.target_count - current_total
        
        if remaining > 0:
            print(f"➕ 需要额外生成 {remaining} 张...")
            extra_count = 0
            augmentation_types = ['light', 'medium', 'heavy']
            
            for i in range(remaining):
                # 随机选择图像
                img_path, label_path = random.choice(valid_pairs)
                base_name = img_path.stem
                aug_type = random.choice(augmentation_types)
                
                success = self.augment_image(img_path, label_path, aug_type, f"extra_{i+1:04d}_{base_name}")
                if success:
                    extra_count += 1
                    # 每生成100张检查是否已达标
                    if (copied_count + base_count + mosaic_count + pair_total + extra_count) >= self.target_count:
                        break
                
                if (i + 1) % 50 == 0:
                    print(f"  已生成额外增强 {i+1}/{remaining} 张")
            
            print(f"✅ 额外增强完成: {extra_count} 张")
        else:
            extra_count = 0
        
        # 6. 最终校验（确保图像和标注一一对应）
        self._final_verification()

    def _final_verification(self):
        """最终验证增强数据的完整性和有效性"""
        print("\n" + "="*60)
        print("🔍 增强数据最终验证")
        print("="*60)
        
        # 统计输出文件
        images = [f for f in self.output_img_dir.glob('*.*') 
                 if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        labels = [f for f in self.output_label_dir.glob('*.txt')]
        
        # 提取文件名（不含扩展名）
        img_names = {f.stem for f in images}
        label_names = {f.stem for f in labels}
        
        # 检查匹配性
        missing_labels = img_names - label_names  # 有图像无标注
        missing_images = label_names - img_names  # 有标注无图像
        
        print(f"总图像数: {len(images)}")
        print(f"总标注数: {len(labels)}")
        
        if not missing_labels and not missing_images:
            print("✅ 图像和标注完全匹配")
        else:
            if missing_labels:
                print(f"❌ 有 {len(missing_labels)} 张图像缺少标注，已自动删除")
                for name in missing_labels:
                    for ext in ['.jpg', '.jpeg', '.png']:
                        img_path = self.output_img_dir / f"{name}{ext}"
                        if img_path.exists():
                            os.remove(img_path)
            if missing_images:
                print(f"❌ 有 {len(missing_images)} 个标注缺少图像，已自动删除")
                for name in missing_images:
                    label_path = self.output_label_dir / f"{name}.txt"
                    if label_path.exists():
                        os.remove(label_path)
        
        # 重新统计
        final_images = len([f for f in self.output_img_dir.glob('*.*') 
                          if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        final_labels = len(list(self.output_label_dir.glob('*.txt')))
        
        print("\n" + "="*60)
        print("🎉 增强完成统计")
        print("="*60)
        print(f"原始数据: {len([f for f in images if f.stem.startswith('original_')])} 张")
        print(f"基础增强: {len([f for f in images if any(t in f.stem for t in ['light', 'medium', 'heavy']) and 'extra' not in f.stem])} 张")
        print(f"马赛克增强: {len([f for f in images if 'mosaic' in f.stem])} 张")
        print(f"双图拼接-横: {len([f for f in images if 'pair_h' in f.stem])} 张")
        print(f"双图拼接-纵: {len([f for f in images if 'pair_v' in f.stem])} 张")
        print(f"额外增强: {len([f for f in images if 'extra' in f.stem])} 张")
        tc = self.target_count
        print(f"最终总数: {final_images} 张（目标：{tc} 张）")
        
        if final_images >= tc:
            print("✅ 成功达到目标数量！")
        else:
            print(f"⚠️  未达到目标，当前 {final_images}/{tc} 张")
        
        print("="*60)


def main():
    # 配置路径（可根据实际情况修改）
    IMAGE_DIR = r"D:\Galaxy\其他\桌面\有羊毛照片"
    LABEL_DIR = r"D:\Galaxy\其他\桌面\有羊毛标注集输出"
    OUTPUT_DIR = r"D:\Galaxy\其他\桌面\augmented_1000_fixed"
    
    # 单类检测写 [0]；多类写 [0,1,...]；不校验类别则写 None（勿写整数 0，会被当成「假」而误关校验）
    VALID_CLASS_IDS = [0]
    # 最小边界框相对面积（过小易混入噪声；小目标约 0.00005～0.0001）
    MIN_BBOX_AREA = 0.00005
    # 生成总张数目标（复制原图 + 基础增强 + 马赛克 + 双图拼接 + 额外增强，合计约到此数）
    TARGET_COUNT = 1000
    # 小目标：明显框阈值（与构造器默认一致，仍偏严时可继续调低）
    PROMINENT_MIN_AREA = 0.0001
    PROMINENT_MIN_SIDE = 0.006
    BBOX_MIN_VISIBILITY = 0.22
    
    print("=" * 60)
    print("YOLO数据增强工具（增强版）")
    print("=" * 60)
    
    # 检查输入目录
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ 错误: 图像目录不存在 {IMAGE_DIR}")
        return
    if not os.path.exists(LABEL_DIR):
        print(f"❌ 错误: 标注目录不存在 {LABEL_DIR}")
        return
    
    # 创建增强器
    augmentor = FixedYOLOAugmentor(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        output_dir=OUTPUT_DIR,
        min_bbox_area=MIN_BBOX_AREA,
        class_ids=VALID_CLASS_IDS,
        use_vertical_flip=False,
        bbox_min_visibility=BBOX_MIN_VISIBILITY,
        require_prominent_labels=True,
        prominent_min_area=PROMINENT_MIN_AREA,
        prominent_min_side=PROMINENT_MIN_SIDE,
        target_count=TARGET_COUNT,
    )
    
    # 开始增强
    start_time = time.time()
    augmentor.augment_to_1000()
    end_time = time.time()
    
    # 显示耗时
    duration = end_time - start_time
    print(f"⏱️  总耗时: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
    print(f"💾 输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()