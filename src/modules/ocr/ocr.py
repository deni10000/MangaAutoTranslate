import os
import re
import cv2
# import jaconv
from transformers import ViTImageProcessor, BertJapaneseTokenizer, VisionEncoderDecoderModel, AutoTokenizer
from PIL import Image
import torch
import numpy as np
from typing import Optional

from huggingface_hub import snapshot_download
from src.modules.base import BaseModule
from src.core.context import MangaPage, TextBlock


class BaseOcr(BaseModule):
    def process(self, page: MangaPage) -> MangaPage:
        valid_crops = []
        valid_indices = []

        for i, blk in enumerate(page.text_blocks):
            x1, y1, x2, y2 = blk.bbox

            if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= page.original_image.shape[1] and y2 <= \
                    page.original_image.shape[0]:
                cropped_img = page.original_image[y1:y2, x1:x2]

                cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cropped_rgb).convert("L").convert("RGB")

                valid_crops.append(pil_img)
                valid_indices.append(i)

        if valid_crops:
            texts = self(valid_crops)

            for idx, text in zip(valid_indices, texts):
                page.text_blocks[idx].source_text = text

        return page

    def __call__(self, img_or_list):
        raise NotImplementedError("Subclasses must implement __call__ method")


class MangaOcr(BaseOcr):
    def __init__(
            self,
            repo_id: str = "kha-white/manga-ocr-base",
            device: Optional[str] = None,
            models_dir: str = "models",
    ):
        local_path = os.path.join(models_dir, "manga_ocr_model")
        marker_file = os.path.join(local_path, ".completed")

        if not (os.path.exists(local_path) and os.path.exists(marker_file)):
            os.makedirs(local_path, exist_ok=True)
            ViTImageProcessor.from_pretrained(repo_id).save_pretrained(local_path)
            BertJapaneseTokenizer.from_pretrained(repo_id).save_pretrained(local_path)
            VisionEncoderDecoderModel.from_pretrained(repo_id).save_pretrained(local_path)

            with open(marker_file, 'w') as f:
                f.write("completed")

        self.processor = ViTImageProcessor.from_pretrained(local_path)
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(local_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(local_path)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    def __call__(self, img_or_list):
        if isinstance(img_or_list, (np.ndarray, Image.Image)):
            imgs = [img_or_list]
        else:
            imgs = img_or_list

        if not imgs:
            return []

        x = self.processor(imgs, return_tensors="pt").pixel_values.to(self.device)

        with torch.inference_mode():
            generated = self.model.generate(x)

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        results = [self._post_process(text) for text in decoded]

        if isinstance(img_or_list, (np.ndarray, Image.Image)):
            return results[0]
        return results

    @staticmethod
    def _post_process(text):
        text = ''.join(text.split())
        text = text.replace('！', '!')
        text = text.replace('…', '...')
        text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
        # text = jaconv.h2z(text, ascii=True, digit=True)
        return text


class EasyOcr(BaseOcr):
    def __init__(
            self,
            lang: str = "en",
            device: Optional[str] = None,
            models_dir: str = "models",
    ):
        import easyocr

        if device is None:
            gpu = torch.cuda.is_available()
        else:
            gpu = torch.device(device).type == "cuda"

        model_dir = os.path.join(models_dir, "easyocr")
        os.makedirs(model_dir, exist_ok=True)

        self.reader = easyocr.Reader(
            [lang],
            gpu=gpu,
            model_storage_directory=model_dir,
        )

    def __call__(self, img_or_list):
        if isinstance(img_or_list, (np.ndarray, Image.Image)):
            imgs = [img_or_list]
            single = True
        else:
            imgs = img_or_list
            single = False

        if not imgs:
            return []

        results = []
        for img in imgs:
            img_np = np.array(img) if isinstance(img, Image.Image) else img
            result = self.reader.readtext(img_np, detail=0)
            text = " ".join(result)
            results.append(text)

        return results[0] if single else results


class PaddleOcr(BaseOcr):
    def __init__(
            self,
            lang: str = "japan",
            device: Optional[str] = None,
            models_dir: str = "models",
    ):
        from paddleocr import PaddleOCR
        self.lang = lang

        det_repo_id = "PaddlePaddle/PP-OCRv5_server_det_safetensors"
        det_model_name = 'PP-OCRv5_server_det'
        det_folder = "paddleocr_det"

        if lang == "korean":
            rec_repo_id = "PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors"
            rec_folder = "paddleocr_korean_rec"
            rec_model_name = 'korean_PP-OCRv5_mobile_rec'
        else:
            rec_repo_id = "PaddlePaddle/PP-OCRv5_server_rec_safetensors"
            rec_folder = "paddleocr_rec"
            rec_model_name = 'PP-OCRv5_server_rec'

        det_dir = self._download_model(det_repo_id, det_folder, models_dir)
        rec_dir = self._download_model(rec_repo_id, rec_folder, models_dir)

        self.ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
            engine="transformers",
            text_detection_model_dir=det_dir,
            text_detection_model_name=det_model_name,
            text_recognition_model_dir=rec_dir,
            text_recognition_model_name=rec_model_name,
            text_det_thresh=0.2,
            text_det_box_thresh=0.2,
            text_det_unclip_ratio=1.1
        )

    @staticmethod
    def _download_model(repo_id: str, folder_name: str, models_dir: str) -> str:
        local_path = os.path.join(models_dir, folder_name)
        marker_file = os.path.join(local_path, ".completed")

        if not (os.path.exists(local_path) and os.path.exists(marker_file)):
            os.makedirs(local_path, exist_ok=True)
            snapshot_download(repo_id=repo_id, local_dir=local_path)
            with open(marker_file, 'w') as f:
                f.write("completed")

        return local_path

    def __call__(self, img_or_list):
        if isinstance(img_or_list, (np.ndarray, Image.Image)):
            imgs = [img_or_list]
            single = True
        else:
            imgs = img_or_list
            single = False

        if not imgs:
            return []

        results = []
        for img in imgs:
            img_np = np.array(img) if isinstance(img, Image.Image) else img
            result = self.ocr.predict(img_np)
            rec_boxes = result[0]["rec_boxes"]

            sorted_indices = sorted(range(len(rec_boxes)),  key=lambda i: (rec_boxes[i][1] + rec_boxes[i][3]) / 2)
            sorted_texts = [result[0]["rec_texts"][i] for i in sorted_indices]

            text = " ".join(sorted_texts)

            print(text)
            results.append(self._post_process(text, self.lang))

        return results[0] if single else results

    @staticmethod
    def _post_process(text: str, lang: str) -> str:
        if lang in ("ja", "ch", "korean"):
            return "".join(text.split())
        return " ".join(text.split())

