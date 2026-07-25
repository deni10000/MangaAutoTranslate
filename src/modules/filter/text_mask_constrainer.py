import cv2
import numpy as np
from src.modules.base import BaseModule
from src.core.context import MangaPage


class TextMaskConstrainer(BaseModule):
    def __init__(
            self,
            max_bubble_shrink_margin: int = 5,
            shrink_ratio: float = 0.05,
    ):
        self.max_bubble_shrink_margin = max_bubble_shrink_margin
        self.shrink_ratio = shrink_ratio

    def shrink_mask(self, mask: np.ndarray, margin: int = 2) -> np.ndarray:
        if margin <= 0 or mask is None:
            return mask
        kernel_size = margin * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.erode(mask, kernel)

    def process(self, page: MangaPage) -> MangaPage:
        if page.mask is None or not np.any(page.mask):
            return page

        if not page.text_blocks:
            return page

        img_h, img_w = page.original_image.shape[:2]
        constraint_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        for tb in page.text_blocks:
            x1, y1, x2, y2 = tb.bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            text_box_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            text_box_mask[y1:y2, x1:x2] = 255

            matched_bubble = tb.bubble

            if matched_bubble is not None and matched_bubble.mask is not None:
                bx1, by1, bx2, by2 = matched_bubble.bbox
                full_b_mask = np.zeros((img_h, img_w), dtype=np.uint8)

                h_crop, w_crop = matched_bubble.mask.shape[:2]
                full_b_mask[by1:by1 + h_crop, bx1:bx1 + w_crop] = matched_bubble.mask

                shrink_margin = min(round((h_crop * w_crop) ** 0.5 * self.shrink_ratio), self.max_bubble_shrink_margin)
                shrunk_b_mask = self.shrink_mask(full_b_mask, margin=shrink_margin)

                allowed_region = cv2.bitwise_and(shrunk_b_mask, text_box_mask)
            else:
                allowed_region = text_box_mask

            constraint_mask = cv2.bitwise_or(constraint_mask, allowed_region)

        page.mask = cv2.bitwise_and(page.mask, constraint_mask)
        return page