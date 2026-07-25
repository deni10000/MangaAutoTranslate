import re
import numpy as np
from src.core.context import MangaPage


class EmptyElementsFilter:
    def __init__(self):
        self.alphabet_re = re.compile(r"[^\W\d_]")

    def process(self, page: MangaPage) -> MangaPage:
        if not page.text_blocks:
            return page

        valid_blocks = []
        valid_bubbles = []
        removed_bboxes = []

        for block in page.text_blocks:
            text = block.source_text or block.translated_text or ""

            if self.alphabet_re.search(text):
                valid_blocks.append(block)
                if block.bubble is not None:
                    if block.bubble not in valid_bubbles:
                        valid_bubbles.append(block.bubble)
            else:
                removed_bboxes.append(block.bbox)

        if removed_bboxes and page.mask is not None:
            for x1, y1, x2, y2 in removed_bboxes:
                page.mask[y1:y2, x1:x2] = 0

        page.text_blocks = valid_blocks
        page.bubbles = valid_bubbles

        return page