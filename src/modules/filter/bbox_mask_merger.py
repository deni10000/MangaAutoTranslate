import numpy as np
from collections import defaultdict

from src.modules.base import BaseModule
from src.core.context import MangaPage, Bubble, TextBlock
from src.core.utils import merge_boxes


class BBoxMaskMerger(BaseModule):
    def process(self, page: MangaPage) -> MangaPage:
        if not page.bubbles and not page.text_blocks:
            return page

        img_h, img_w = page.original_image.shape[:2]

        raw_bubbles = []
        for b in page.bubbles:
            x1, y1, x2, y2 = b.bbox
            full_mask = None

            if b.mask is not None:
                full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                mh, mw = b.mask.shape[:2]
                full_mask[y1:y1 + mh, x1:x1 + mw] = b.mask

            raw_bubbles.append((x1, y1, x2, y2, full_mask))

        raw_texts = [(t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3], None) for t in page.text_blocks]

        merged_bubbles_raw = merge_boxes(raw_bubbles)
        merged_texts_raw = merge_boxes(raw_texts)

        new_bubbles = []
        for x1, y1, x2, y2, full_mask in merged_bubbles_raw:
            crop_mask = None
            if full_mask is not None:
                cropped = full_mask[y1:y2, x1:x2]
                if cropped.size > 0 and np.any(cropped):
                    crop_mask = cropped

            new_bubbles.append(Bubble(bbox=(x1, y1, x2, y2), mask=crop_mask))

        bubble_to_texts = defaultdict(list)
        unassigned_texts = []

        for tb in merged_texts_raw:
            tb_bbox = (tb[0], tb[1], tb[2], tb[3])
            cx = (tb[0] + tb[2]) // 2
            cy = (tb[1] + tb[3]) // 2

            matched_bubble = None
            for b in new_bubbles:
                bx1, by1, bx2, by2 = b.bbox
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    matched_bubble = b
                    break

            if matched_bubble:
                bubble_to_texts[matched_bubble].append(tb_bbox)
            else:
                unassigned_texts.append(tb_bbox)

        new_text_blocks = []

        for bbox in unassigned_texts:
            new_text_blocks.append(TextBlock(bbox=bbox, bubble=None))

        for bubble, bboxes in bubble_to_texts.items():
            min_x = min(b[0] for b in bboxes)
            min_y = min(b[1] for b in bboxes)
            max_x = max(b[2] for b in bboxes)
            max_y = max(b[3] for b in bboxes)

            new_text_blocks.append(TextBlock(
                bbox=(min_x, min_y, max_x, max_y),
                bubble=bubble
            ))

        page.bubbles = new_bubbles
        page.text_blocks = new_text_blocks

        return page