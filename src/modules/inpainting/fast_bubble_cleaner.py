import cv2
import numpy as np

from src.modules.base import BaseModule
from src.core.context import MangaPage


class FastBubbleCleaner(BaseModule):
    def __init__(
            self,
            spread_threshold: float = 20.0,
            kernel_size: int = 3,
            percentile_margin: int = 5,
    ):
        self.spread_threshold = spread_threshold
        self.kernel_size = kernel_size
        self.percentile_margin = percentile_margin

    def process(self, page: MangaPage) -> MangaPage:
        if page.mask is None or not np.any(page.mask):
            return page

        if page.inpainted_image is None:
            page.inpainted_image = page.original_image.copy()

        img = page.inpainted_image
        cleaned_mask = page.mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))

        processed_bubbles = set()

        for block in page.text_blocks:
            bubble = block.bubble
            if bubble is None or bubble in processed_bubbles:
                continue
            processed_bubbles.add(bubble)

            x1, y1, x2, y2 = map(int, bubble.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            img_crop = img[y1:y2, x1:x2]
            page_mask_crop = page.mask[y1:y2, x1:x2]

            if bubble.mask is not None:
                if bubble.mask.shape[:2] == img.shape[:2]:
                    b_mask = bubble.mask[y1:y2, x1:x2]
                else:
                    b_mask = bubble.mask

                text_mask = cv2.bitwise_and(page_mask_crop, (b_mask > 0).astype(np.uint8) * 255)
            else:
                text_mask = page_mask_crop.copy()

            if not np.any(text_mask):
                continue

            dilated_mask = cv2.dilate(text_mask, kernel, iterations=1)

            contour_mask = cv2.subtract(dilated_mask, text_mask)

            boundary_pixels = img_crop[contour_mask > 0]
            if len(boundary_pixels) == 0:
                continue

            lo = np.percentile(boundary_pixels, self.percentile_margin, axis=0)
            hi = np.percentile(boundary_pixels, 100 - self.percentile_margin, axis=0)
            spread = np.mean(hi - lo)

            if spread < self.spread_threshold:
                fill_color = np.median(boundary_pixels, axis=0).astype(np.uint8)
                img_crop[dilated_mask > 0] = fill_color

                cleaned_mask[y1:y2, x1:x2][dilated_mask > 0] = 0

        page.mask = cleaned_mask
        return page