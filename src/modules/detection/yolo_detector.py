import os
import torch
import numpy as np
import cv2
from typing import Optional
from collections import defaultdict
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from src.core.utils import merge_boxes
from src.modules.base import BaseModule
from src.core.context import MangaPage, Bubble, TextBlock, Panel


class YoloV26SegmentationDetector(BaseModule):
    def __init__(
            self,
            repo_id_1: str = "ShadowB/Manga109-panel-balloon-text-yolov26-segmentation",
            filename_1: str = "best.pt",
            repo_id_2: str = "deni1000/yolo26-text-bubble-detection",
            filename_2: str = "best_small.pt",
            device: Optional[str] = None,
            conf_threshold: float = 0.05,
            text_mask_margin: int = 4,
            final_mask_margin: int = 3,
            min_text_area: int = 8,
            min_text_width: int = 2,
            min_text_height: int = 2,
            max_imgsz: int = 1280,
            min_fill_ratio: float = 0.15,
            models_dir: str = "models",
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.conf_threshold = conf_threshold
        self.text_mask_margin = text_mask_margin
        self.final_mask_margin = final_mask_margin
        self.min_text_area = min_text_area
        self.min_text_width = min_text_width
        self.min_text_height = min_text_height
        self.max_imgsz = max_imgsz
        self.min_fill_ratio = min_fill_ratio
        self.models_dir = models_dir

        model_path_1 = self._download_model(repo_id_1, filename_1, "yolov26_seg_model")
        self.model_1 = YOLO(model_path_1)

        model_path_2 = self._download_model(repo_id_2, filename_2, "yolov26_det_model")
        self.model_2 = YOLO(model_path_2)

    def _download_model(self, repo_id: str, filename: str, folder_name: str) -> str:
        local_path = os.path.join(self.models_dir, folder_name)
        marker_file = os.path.join(local_path, ".completed")

        if not (os.path.exists(local_path) and os.path.exists(marker_file)):
            os.makedirs(local_path, exist_ok=True)
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_path)
            with open(marker_file, 'w') as f:
                f.write("completed")

        return os.path.join(local_path, filename)

    def expand_mask(self, mask: np.ndarray, margin: int = 5) -> np.ndarray:
        if margin <= 0 or mask is None:
            return mask
        kernel_size = margin * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(mask, kernel)

    def _resolve_imgsz(self, img_h: int, img_w: int):
        return (img_h, img_w) if self.max_imgsz > max(img_h, img_w) else self.max_imgsz

    def process(self, page: MangaPage) -> MangaPage:
        img_h, img_w, _ = page.original_image.shape
        imgsz = self._resolve_imgsz(img_h, img_w)

        results_1 = self.model_1.predict(
            source=page.original_image,
            device=str(self.device),
            verbose=False,
            conf=self.conf_threshold,
            retina_masks=True,
            imgsz=imgsz,
            rect=True
        )[0]

        classes_1 = results_1.boxes.cls.cpu().numpy().astype(int) if len(results_1.boxes) > 0 else []
        masks_data_1 = results_1.masks.data if results_1.masks is not None else None

        global_bubble_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        text_mask_indices_1 = []

        raw_panels = []

        for idx, cls_id in enumerate(classes_1):
            if cls_id == 2 and masks_data_1 is not None:
                b_mask = (masks_data_1[idx] > 0).cpu().numpy().astype(np.uint8) * 255
                global_bubble_mask = cv2.bitwise_or(global_bubble_mask, b_mask)

            elif cls_id == 1:
                text_mask_indices_1.append(idx)

            else:
                raw_panel = tuple(map(int, results_1.boxes.xyxy[idx].cpu()))
                raw_panels.append((raw_panel[0], raw_panel[1], raw_panel[2], raw_panel[3], None))

        seg_text_boxes = []

        if text_mask_indices_1 and masks_data_1 is not None:
            combined_text_mask = torch.any(
                masks_data_1[text_mask_indices_1] > 0, dim=0
            )
            raw_text_mask = (combined_text_mask.cpu().numpy() * 255).astype(np.uint8)

            raw_text_mask = self.expand_mask(raw_text_mask, margin=self.text_mask_margin)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                raw_text_mask, connectivity=8
            )

            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]

                if area >= self.min_text_area and w >= self.min_text_width and h >= self.min_text_height:
                    seg_text_boxes.append((x, y, x + w, y + h, None))

            page.mask = self.expand_mask(raw_text_mask, margin=self.final_mask_margin)
        else:
            page.mask = np.zeros((img_h, img_w), dtype=np.uint8)

        if page.original_image.ndim == 3 and page.original_image.shape[2] == 3:
            gray_image = cv2.cvtColor(page.original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = page.original_image

        results_2 = self.model_2.predict(
            source=gray_image,
            device=str(self.device),
            verbose=False,
            conf=self.conf_threshold,
            retina_masks=False,
            imgsz=imgsz,
            rect=True,
            iou=1,
        )[0]

        boxes_2 = results_2.boxes.xyxy.cpu().numpy() if len(results_2.boxes) > 0 else []
        classes_2 = results_2.boxes.cls.cpu().numpy().astype(int) if len(results_2.boxes) > 0 else []

        raw_bubbles = []
        raw_texts = []

        for box, cls_id in zip(boxes_2, classes_2):
            x1, y1, x2, y2 = map(int, box)

            if cls_id == 0:
                raw_bubbles.append((x1, y1, x2, y2, None))
            elif cls_id == 1:
                raw_texts.append((x1, y1, x2, y2, None))

        raw_texts.extend(seg_text_boxes)

        merged_texts = merge_boxes(raw_texts)
        merged_bubbles = merge_boxes(raw_bubbles)
        merged_panels = merge_boxes(raw_panels)

        panels = []
        for idx, (x1, y1, x2, y2, _) in enumerate(merged_panels):
            x1, x2 = max(0, x1), min(img_w, x2)
            y1, y2 = max(0, y1), min(img_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            panels.append(Panel(bbox=(x1, y1, x2, y2)))

        page.panels = panels

        assignment_map = np.full((img_h, img_w), -1, dtype=np.int32)
        dist_map = np.full((img_h, img_w), np.inf, dtype=np.float32)

        y_indices, x_indices = np.indices((img_h, img_w))

        for idx, (x1, y1, x2, y2, _) in enumerate(merged_bubbles):
            x1, x2 = max(0, x1), min(img_w, x2)
            y1, y2 = max(0, y1), min(img_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            slice_y = slice(y1, y2)
            slice_x = slice(x1, x2)

            x_grid = x_indices[slice_y, slice_x]
            y_grid = y_indices[slice_y, slice_x]

            dist_to_edges = np.minimum(
                np.minimum(x_grid - x1, x2 - x_grid),
                np.minimum(y_grid - y1, y2 - y_grid)
            )
            dist_val = -dist_to_edges.astype(np.float32)

            closer_mask = dist_val < dist_map[slice_y, slice_x]

            dist_map[slice_y, slice_x][closer_mask] = dist_val[closer_mask]
            assignment_map[slice_y, slice_x][closer_mask] = idx

        bubbles = []
        for idx, (x1, y1, x2, y2, _) in enumerate(merged_bubbles):
            x1, x2 = max(0, x1), min(img_w, x2)
            y1, y2 = max(0, y1), min(img_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            b_mask_full = (assignment_map == idx) & (global_bubble_mask > 0)
            crop_mask = (b_mask_full[y1:y2, x1:x2] * 255).astype(np.uint8)
            if crop_mask.size > 0:
                fill_ratio = np.count_nonzero(crop_mask) / crop_mask.size
                if fill_ratio < self.min_fill_ratio:
                    crop_mask = None
            else:
                crop_mask = None

            bubbles.append(Bubble(bbox=(x1, y1, x2, y2), mask=crop_mask))

        bubble_to_texts = defaultdict(list)
        unassigned_texts = []

        for x1, y1, x2, y2, _ in merged_texts:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            matched_bubble = None
            for b in bubbles:
                bx1, by1, bx2, by2 = b.bbox
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    matched_bubble = b
                    break

            if matched_bubble:
                bubble_to_texts[matched_bubble].append((x1, y1, x2, y2))
            else:
                unassigned_texts.append((x1, y1, x2, y2))

        text_blocks = []

        for b in unassigned_texts:
            text_blocks.append(TextBlock(bbox=b, bubble=None))

        for bubble, bboxes in bubble_to_texts.items():
            min_x = min(b[0] for b in bboxes)
            min_y = min(b[1] for b in bboxes)
            max_x = max(b[2] for b in bboxes)
            max_y = max(b[3] for b in bboxes)

            text_blocks.append(TextBlock(
                bbox=(min_x, min_y, max_x, max_y),
                bubble=bubble
            ))

        page.bubbles.extend(bubbles)
        page.text_blocks.extend(text_blocks)

        return page