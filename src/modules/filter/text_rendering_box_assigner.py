from typing import Optional, Tuple
import cv2
import numpy as np
from src.core.context import MangaPage, TextBlock


class TextRenderingBoxAssigner:

    def __init__(
            self,
            bubble_padding_ratio: float = 0.05,
            variance_threshold: float = 600.0,
            fixed_threshold: int = 20,
            erode_kernel_size: int = 3,
            flood_fill_diff: int = 20,
            initial_box_ratio: float = 0.7,
            min_fill_ratio: float = 0.15,
            min_rendering_size: float = 10.0,
    ):
        self.bubble_padding_ratio = bubble_padding_ratio
        self.variance_threshold = variance_threshold
        self.fixed_threshold = fixed_threshold
        self.erode_kernel_size = erode_kernel_size
        self.flood_fill_diff = flood_fill_diff
        self.initial_box_ratio = initial_box_ratio
        self.min_fill_ratio = min_fill_ratio
        self.min_rendering_size = min_rendering_size

    def get_outer_contour_by_point(
            self,
            crop_img: np.ndarray,
            point: Tuple[int, int],
            text_bbox: Tuple[int, int, int, int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        x, y = point
        h_img, w_img = crop_img.shape[:2]
        if x < 0 or y < 0 or y >= h_img or x >= w_img:
            return None, None

        if len(crop_img.shape) == 3:
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop_img.copy()

        x_min, y_min, x_max, y_max = text_bbox

        x1, y1 = max(0, int(x_min)), max(0, int(y_min))
        x2, y2 = min(w_img, int(x_max)), min(h_img, int(y_max))

        text_region = gray[y1:y2, x1:x2]

        variance = float(np.var(text_region))

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.erode_kernel_size, self.erode_kernel_size)
        )
        thickened_gray = cv2.erode(gray, kernel, iterations=1)

        if variance < self.variance_threshold:
            ff_mask = np.zeros((h_img + 2, w_img + 2), dtype=np.uint8)
            flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
            cv2.floodFill(
                thickened_gray, ff_mask, (x, y), 255,
                loDiff=self.flood_fill_diff, upDiff=self.flood_fill_diff, flags=flags
            )
            binary = ff_mask[1:-1, 1:-1]
        else:
            _, binary = cv2.threshold(
                thickened_gray, self.fixed_threshold, 255, cv2.THRESH_BINARY
            )

        if binary[y, x] == 0:
            binary = 255 - binary

        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=4
        )
        component_label = labels[y, x]

        if component_label == 0:
            return None, None

        raw_mask = (labels == component_label).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, None

        contour = max(contours, key=cv2.contourArea)

        filled_mask = np.zeros_like(raw_mask)
        cv2.drawContours(filled_mask, [contour], -1, 255, thickness=-1)

        return contour, filled_mask

    @staticmethod
    def get_inscribed_rectangle(
            init_params,
            dist_outside,
            lr_start=0.5,
            epochs=200,
            lambda_penalty=1000.0,
            num_samples=32,
            max_norm=0.05,
    ):
        h_img, w_img = dist_outside.shape

        scales = np.array([w_img, h_img, w_img, h_img, 180.0], dtype=np.float32)
        offsets = np.array([0.0, 0.0, 0.0, 0.0, -90.0], dtype=np.float32)

        init_params_norm = (
                np.array(init_params, dtype=np.float32) - offsets
        ) / scales
        params_norm = init_params_norm.copy()

        min_w_norm = 10.0 / w_img
        min_h_norm = 10.0 / h_img

        local_pts = []
        for sx in [-0.5, 0.5]:
            for sy in [-0.5, 0.5]:
                local_pts.append([sx, sy])
        for t in np.linspace(-0.45, 0.45, num_samples):
            local_pts.append([t, -0.5])
            local_pts.append([t, 0.5])
            local_pts.append([-0.5, t])
            local_pts.append([0.5, t])

        local_pts = np.array(local_pts, dtype=np.float32)
        s_x = local_pts[:, 0]
        s_y = local_pts[:, 1]

        for epoch in range(1, epochs + 1):
            params_pixel = params_norm * scales + offsets
            cx, cy, w, h, angle = params_pixel

            rad = angle * np.pi / 180.0
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)

            x_local = s_x * w
            y_local = s_y * h

            pts_x = cx + x_local * cos_a - y_local * sin_a
            pts_y = cy + x_local * sin_a + y_local * cos_a

            pts_x_clipped = np.clip(pts_x, 0, w_img - 1.001)
            pts_y_clipped = np.clip(pts_y, 0, h_img - 1.001)

            x0 = np.floor(pts_x_clipped).astype(np.int32)
            x1 = x0 + 1
            y0 = np.floor(pts_y_clipped).astype(np.int32)
            y1 = y0 + 1

            dx = pts_x_clipped - x0
            dy = pts_y_clipped - y0

            wa = (1.0 - dx) * (1.0 - dy)
            wb = dx * (1.0 - dy)
            wc = (1.0 - dx) * dy
            wd = dx * dy

            d00 = dist_outside[y0, x0]
            d01 = dist_outside[y0, x1]
            d10 = dist_outside[y1, x0]
            d11 = dist_outside[y1, x1]

            distances = wa * d00 + wb * d01 + wc * d10 + wd * d11

            dD_dx = -(1.0 - dy) * d00 + (1.0 - dy) * d01 - dy * d10 + dy * d11
            dD_dy = -(1.0 - dx) * d00 - dx * d01 + (1.0 - dx) * d10 + dx * d11

            in_bounds_x = (pts_x >= 0) & (pts_x <= w_img - 1.001)
            in_bounds_y = (pts_y >= 0) & (pts_y <= h_img - 1.001)
            dD_dx = np.where(in_bounds_x, dD_dx, 0.0)
            dD_dy = np.where(in_bounds_y, dD_dy, 0.0)

            dPenalty_dx = 2.0 * distances * dD_dx
            dPenalty_dy = 2.0 * distances * dD_dy

            out_left = np.maximum(0.0, -pts_x)
            out_right = np.maximum(0.0, pts_x - (w_img - 1))
            out_top = np.maximum(0.0, -pts_y)
            out_bottom = np.maximum(0.0, pts_y - (h_img - 1))

            dBound_dx = -2.0 * out_left + 2.0 * out_right
            dBound_dy = -2.0 * out_top + 2.0 * out_bottom

            g_x = lambda_penalty * (dPenalty_dx + dBound_dx)
            g_y = lambda_penalty * (dPenalty_dy + dBound_dy)

            grad_cx = np.sum(g_x)
            grad_cy = np.sum(g_y)

            grad_w = -h + np.sum(g_x * s_x * cos_a + g_y * s_x * sin_a)
            grad_h = -w + np.sum(-g_x * s_y * sin_a + g_y * s_y * cos_a)

            grad_rad = np.sum(
                g_x * (-x_local * sin_a - y_local * cos_a)
                + g_y * (x_local * cos_a - y_local * sin_a)
            )
            grad_angle = grad_rad * (np.pi / 180.0)

            grad_pixel = np.array(
                [grad_cx, grad_cy, grad_w, grad_h, grad_angle], dtype=np.float32
            )

            grad_norm = grad_pixel * scales

            grad_norm_l2 = np.linalg.norm(grad_norm)
            if grad_norm_l2 > max_norm:
                grad_norm = grad_norm * (max_norm / grad_norm_l2)

            current_lr = lr_start * (1.0 - (epoch - 1) / epochs)
            params_norm -= current_lr * grad_norm

            params_norm[0] = np.clip(params_norm[0], 0.0, 1.0)
            params_norm[1] = np.clip(params_norm[1], 0.0, 1.0)
            params_norm[2] = np.clip(params_norm[2], min_w_norm, 1.0)
            params_norm[3] = np.clip(params_norm[3], min_h_norm, 1.0)
            params_norm[4] = params_norm[4] % 1.0

        return params_norm * scales + offsets

    def normalize_rect(
            self, w: float, h: float, angle: float
    ) -> Tuple[float, float, float]:
        while angle > 90:
            angle -= 180
        while angle <= -90:
            angle += 180

        if angle > 45:
            angle -= 90
            w, h = h, w
        elif angle < -45:
            angle += 90
            w, h = h, w

        return w, h, angle

    def process(self, page: MangaPage) -> MangaPage:
        working_img = (
            page.inpainted_image
            if page.inpainted_image is not None
            else page.original_image
        )
        h_img, w_img = working_img.shape[:2]

        for block in page.text_blocks:
            x_min, y_min, x_max, y_max = block.bbox
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            w = float(x_max - x_min)
            h = float(y_max - y_min)
            angle = 0.0

            bubble = block.bubble
            if bubble is not None:
                bx_min, by_min, bx_max, by_max = bubble.bbox
                cx = (bx_min + bx_max) / 2.0
                cy = (by_min + by_max) / 2.0
                w = (bx_max - bx_min) * self.initial_box_ratio
                h = (by_max - by_min) * self.initial_box_ratio

                bx_min_i, by_min_i = max(0, int(bx_min)), max(0, int(by_min))
                bx_max_i, by_max_i = min(w_img, int(bx_max)), min(h_img, int(by_max))

                crop_w = bx_max_i - bx_min_i
                crop_h = by_max_i - by_min_i

                if crop_w > 0 and crop_h > 0:
                    crop_area = crop_w * crop_h
                    local_mask = None

                    if bubble.mask is not None:
                        if bubble.mask.shape[:2] == (h_img, w_img):
                            local_mask = bubble.mask[by_min_i:by_max_i, bx_min_i:bx_max_i]
                        else:
                            local_mask = bubble.mask
                    else:
                        crop = working_img[by_min_i:by_max_i, bx_min_i:bx_max_i]
                        local_point = (crop_w // 2, crop_h // 2)
                        lx_min, ly_min = x_min - bx_min_i, y_min - by_min_i
                        lx_max, ly_max = x_max - bx_min_i, y_max - by_min_i
                        local_bbox = (lx_min, ly_min, lx_max, ly_max)

                        _, local_mask = self.get_outer_contour_by_point(
                            crop, local_point, local_bbox
                        )

                    is_valid_mask = False
                    if local_mask is not None:
                        fill_ratio = np.count_nonzero(local_mask) / crop_area
                        if fill_ratio >= self.min_fill_ratio:
                            is_valid_mask = True

                    if not is_valid_mask:
                        local_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
                        pad_w = int(crop_w * self.min_fill_ratio)
                        pad_h = int(crop_h * self.min_fill_ratio)

                        y1_p, y2_p = pad_h, max(pad_h + 1, crop_h - pad_h)
                        x1_p, x2_p = pad_w, max(pad_w + 1, crop_w - pad_w)

                        local_mask[y1_p:y2_p, x1_p:x2_p] = 255
                        bubble.mask = local_mask

                    contours, _ = cv2.findContours(
                        local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    contour = max(contours, key=cv2.contourArea) if contours else None
                    block.contour = contour

                    if is_valid_mask and contour is not None:
                        dist_outside = cv2.distanceTransform(
                            255 - local_mask, cv2.DIST_L2, 5
                        )

                        lx_min, ly_min = x_min - bx_min_i, y_min - by_min_i
                        lx_max, ly_max = x_max - bx_min_i, y_max - by_min_i

                        cx_text_local = (lx_min + lx_max) / 2.0
                        cy_text_local = (ly_min + ly_max) / 2.0
                        w_text = float(lx_max - lx_min)
                        h_text = float(ly_max - ly_min)

                        initial_guess = [cx_text_local, cy_text_local, w_text, h_text, 0.0]
                        cx_loc, cy_loc, w_opt, h_opt, angle_opt = (
                            self.get_inscribed_rectangle(initial_guess, dist_outside)
                        )

                        cx = cx_loc + bx_min_i
                        cy = cy_loc + by_min_i
                        w = w_opt * (1.0 - self.bubble_padding_ratio)
                        h = h_opt * (1.0 - self.bubble_padding_ratio)
                        angle = angle_opt

            w, h = max(w, self.min_rendering_size), max(h, self.min_rendering_size)
            w, h, angle = self.normalize_rect(w, h, angle)
            block.rendering_box = (cx, cy, w, h, angle)

        return page