import io
import os
import uuid
from typing import Optional, Union, List, Tuple
import cv2
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from fontTools.ttLib import TTFont as FontToolsTTFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from src.core.context import MangaPage
from fontTools.merge import Merger
import fitz
import numpy as np


class PDFRenderer:

    def __init__(self,
                 stroke_width_ratio: float = 0.06,
                 default_font_path: Optional[Union[str, List[str]]] = None,
                 default_text_color: str = '#000000',
                 default_stroke_color: str = '#FFFFFF',
                 debug: bool = False,
                 fallback_font_paths: Optional[List[str]] = None,
                 jpeg_quality: int = 95,
                 min_size_default: int = 10,
                 leading_ratio: float = 1.1,
                 hyphenation_lang: str = 'ru_RU',
                 font_size_threshold_ratio: float = 1.5):

        self.stroke_width_ratio = stroke_width_ratio
        self.default_text_color = default_text_color
        self.default_stroke_color = default_stroke_color
        self.debug = debug
        self.jpeg_quality = jpeg_quality
        self.min_size_default = min_size_default
        self.leading_ratio = leading_ratio
        self.hyphenation_lang = hyphenation_lang
        self.font_size_threshold_ratio = font_size_threshold_ratio

        self._font_cache = {}
        self.font_name = 'Helvetica-Bold'

        if default_font_path:
            registered_name = self._get_or_register_font(default_font_path)
            if registered_name != 'Helvetica-Bold':
                self.font_name = registered_name

        if self.font_name == 'Helvetica-Bold':
            fallback_paths = fallback_font_paths or [
                r"C:\Windows\Fonts\arialbd.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            ]
            for path in fallback_paths:
                if os.path.exists(path):
                    self.font_name = self._get_or_register_font(path)
                    break

    def _merge_fonts(self, valid_paths):
        cleaned_streams = []

        for path in valid_paths:
            font = FontToolsTTFont(path)

            for tag in ['fvar', 'gvar', 'HVAR', 'VVAR', 'MVAR', 'STAT', 'cvar']:
                if tag in font:
                    del font[tag]

            if 'GDEF' in font and hasattr(font['GDEF'].table, 'VarStore'):
                font['GDEF'].table.VarStore = None

            buf = io.BytesIO()
            font.save(buf)
            buf.seek(0)
            cleaned_streams.append(buf)

        merger = Merger()
        merged_font = merger.merge(cleaned_streams)

        out = io.BytesIO()
        merged_font.save(out)
        out.seek(0)

        return out

    def _get_or_register_font(self, font_input: Optional[Union[str, List[str]]]) -> str:
        if not font_input:
            return self.font_name

        font_paths = [font_input] if isinstance(font_input, str) else list(font_input)
        valid_paths = tuple(p for p in font_paths if p and os.path.exists(p))

        if not valid_paths:
            return self.font_name

        if valid_paths in self._font_cache:
            return self._font_cache[valid_paths]

        unique_name = f"CustomFont_{uuid.uuid4().hex[:8]}"

        if len(valid_paths) == 1:
            pdfmetrics.registerFont(TTFont(unique_name, valid_paths[0]))
        else:
            merged_stream = self._merge_fonts(valid_paths)
            pdfmetrics.registerFont(TTFont(unique_name, merged_stream))

        self._font_cache[valid_paths] = unique_name
        return unique_name

    def _convert_pdf_to_image(self, pdf_bytes: bytes, target_w: int, target_h: int) -> np.ndarray:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pdf_page = doc[0]
        zoom_x = target_w / pdf_page.rect.width
        zoom_y = target_h / pdf_page.rect.height
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = pdf_page.get_pixmap(matrix=mat)

        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        if pix.n == 4:
            return cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            return cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    def is_bad_render(self, paragraph, original_text: str) -> bool:
        bl_para = getattr(paragraph, 'blPara', None)
        if not bl_para or len(bl_para.lines) <= 1:
            return False

        valid_hyphens = ('-', '\xad', '\u2010', '–', '—')

        lines_words = []
        for line in bl_para.lines:
            words = line.words if type(line).__name__ == 'FragLine' else line[1]
            line_words = [getattr(w, 'text', str(w)) for w in words]
            lines_words.append(line_words)

        for i in range(len(lines_words) - 1):
            curr_words = lines_words[i]
            next_words = lines_words[i + 1]
            if not curr_words or not next_words:
                continue

            last_word = curr_words[-1].strip()
            first_word = next_words[0].strip()
            if not last_word or not first_word:
                continue

            if last_word.endswith(valid_hyphens):
                continue

            if last_word.endswith(' '):
                continue

            if first_word.startswith(' '):
                continue

            combined = last_word + first_word
            if combined in original_text:
                return True

        return False

    def _get_perfect_font_size(self, text: str, font_name: str, max_w: float, max_h: float, start_size: int = 40,
                               min_size: int = 8, correct_shift=True) -> int:
        low = min_size
        high = start_size
        best_size = min_size

        while low <= high:
            mid = (low + high) // 2
            temp_style = ParagraphStyle(
                'TempStyle',
                fontName=font_name,
                fontSize=mid,
                leading=mid * self.leading_ratio,
                alignment=TA_CENTER,
                embeddedHyphenation=1,
                hyphenationLang=self.hyphenation_lang,
            )
            p = Paragraph(text, temp_style)
            _, actual_height = p.wrap(max_w, max_h)

            if actual_height <= max_h and (not correct_shift or not self.is_bad_render(p, text)):
                best_size = mid
                low = mid + 1
            else:
                high = mid - 1

        return best_size

    def process(self, page: MangaPage) -> MangaPage:
        working_img = page.inpainted_image if page.inpainted_image is not None else page.original_image
        h_img, w_img = working_img.shape[:2]

        _, img_encoded = cv2.imencode('.jpg', working_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        img_stream = io.BytesIO(img_encoded.tobytes())

        pdf_buffer = io.BytesIO()
        pdf_canvas = canvas.Canvas(pdf_buffer, pagesize=(w_img, h_img))
        pdf_canvas.drawImage(ImageReader(img_stream), 0, 0, width=w_img, height=h_img)

        current_stroke_color = self.default_stroke_color

        original_begin_text = pdf_canvas.beginText

        def patched_begin_text(*args, **kwargs):
            tx = original_begin_text(*args, **kwargs)
            tx.setTextRenderMode(2)
            tx.setStrokeColor(HexColor(current_stroke_color))
            return tx

        pdf_canvas.beginText = patched_begin_text

        for block in page.text_blocks:
            if not block.translated_text:
                continue

            x_min, y_min, x_max, y_max = block.bbox
            w_def = max(float(x_max - x_min), 10.0)
            h_def = max(float(y_max - y_min), 10.0)

            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            w, h, angle = w_def, h_def, 0.0

            rendering_box = getattr(block, 'rendering_box', None)
            if rendering_box is not None:
                cx_rend, cy_rend, w_rend, h_rend, angle_rend = rendering_box
                cx, cy, w, h, angle = cx_rend, cy_rend, w_rend, h_rend, angle_rend

            if self.debug:
                pdf_canvas.saveState()
                pdf_canvas.setStrokeColor(HexColor('#FF0000'))
                pdf_canvas.setLineWidth(1.0)
                pdf_canvas.rect(x_min, h_img - y_max, x_max - x_min, y_max - y_min, stroke=1, fill=0)
                pdf_canvas.restoreState()

                contour = getattr(block, 'contour', None)
                if contour is not None and len(contour) > 1 and block.bubble is not None:
                    pdf_canvas.saveState()
                    pdf_canvas.setStrokeColor(HexColor('#0000FF'))
                    pdf_canvas.setLineWidth(1.5)

                    off_x, off_y = block.bubble.bbox[0], block.bubble.bbox[1]

                    path = pdf_canvas.beginPath()
                    x_start, y_start = contour[0][0]
                    path.moveTo(x_start + off_x, h_img - (y_start + off_y))

                    for pt in contour[1:]:
                        pt_x, pt_y = pt[0]
                        path.lineTo(pt_x + off_x, h_img - (pt_y + off_y))

                    path.close()
                    pdf_canvas.drawPath(path, stroke=1, fill=0)
                    pdf_canvas.restoreState()

            b_font_path = getattr(block, 'font_path', None)
            block_font_name = self._get_or_register_font(b_font_path)

            block_text_color = getattr(block, 'text_color', None) or self.default_text_color
            current_stroke_color = getattr(block, 'stroke_color', None) or self.default_stroke_color

            block_stroke_ratio = getattr(block, 'stroke_width_ratio', None)
            if block_stroke_ratio is None:
                block_stroke_ratio = self.stroke_width_ratio

            if block.font_size is None:
                mn_s = self.min_size_default
                font_size_1 = self._get_perfect_font_size(block.translated_text, block_font_name, w, h, min_size=mn_s)
                font_size_2 = self._get_perfect_font_size(block.translated_text, block_font_name, w, h, min_size=mn_s,
                                                          correct_shift=False)
                block.font_size = font_size_2 if font_size_2 >= font_size_1 * self.font_size_threshold_ratio else font_size_1

            font_size = block.font_size

            actual_stroke_width = font_size * block_stroke_ratio

            manga_style = ParagraphStyle(
                f'MangaStyle_{id(block)}',
                fontName=block_font_name,
                fontSize=font_size,
                leading=font_size * self.leading_ratio,
                textColor=HexColor(block_text_color),
                alignment=TA_CENTER,
                embeddedHyphenation=1,
                hyphenationLang=self.hyphenation_lang,
            )

            paragraph = Paragraph(block.translated_text, manga_style)
            actual_width, actual_height = paragraph.wrap(w, h)

            pdf_canvas.saveState()
            pdf_canvas.translate(cx, h_img - cy)
            pdf_canvas.rotate(-angle)

            if self.debug:
                pdf_canvas.setStrokeColor(HexColor('#00FF00'))
                pdf_canvas.setLineWidth(1.5)
                pdf_canvas.rect(-w / 2.0, -h / 2.0, w, h, stroke=1, fill=0)

            pdf_canvas.setStrokeColor(HexColor(current_stroke_color))
            pdf_canvas.setLineWidth(actual_stroke_width)
            pdf_canvas.setFillColor(HexColor(block_text_color))

            draw_x = -w / 2.0
            draw_y = -actual_height / 2.0
            paragraph.drawOn(pdf_canvas, draw_x, draw_y)

            pdf_canvas.restoreState()

        pdf_canvas.showPage()
        pdf_canvas.save()

        pdf_buffer.seek(0)
        page.pdf_data = pdf_buffer.getvalue()
        page.translated_image = self._convert_pdf_to_image(page.pdf_data, w_img, h_img)

        return page

    def save_to_pdf(self, page: MangaPage, pdf_path: str):
        if page.pdf_data is None:
            raise ValueError("MangaPage has no PDF data.")
        with open(pdf_path, 'wb') as f:
            f.write(page.pdf_data)