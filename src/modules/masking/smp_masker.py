import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download
from typing import Optional

from src.modules.base import BaseModule
from src.core.context import MangaPage, TextBlock


def convert_batchnorm_to_groupnorm(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = 8
            if num_channels < num_groups or num_channels % num_groups != 0:
                for i in range(min(num_channels, 8), 1, -1):
                    if num_channels % i == 0:
                        num_groups = i
                        break
                else:
                    num_groups = 1
            setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
        else:
            convert_batchnorm_to_groupnorm(child)


class SmpTextMasker(BaseModule):
    def __init__(
            self,
            repo_id: str = "a-b-c-x-y-z/Manga-Text-Segmentation-2025",
            filename: str = "model.pth",
            encoder_name: str = "tu-efficientnetv2_rw_m",
            device: Optional[str] = None,
            threshold: float = 0.01,
            fill_holes: bool = False,
            close_gaps_kernel: int = 0,
            padding_iter: int = 2,
            tta_hflip: bool = False,
            tta_vflip: bool = False,
            models_dir: str = "models",
    ):
        local_path = os.path.join(models_dir, "smp_text_masker_model")
        marker_file = os.path.join(local_path, ".completed")

        if not (os.path.exists(local_path) and os.path.exists(marker_file)):
            os.makedirs(local_path, exist_ok=True)
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_path)
            with open(marker_file, 'w') as f:
                f.write("completed")

        model_path = os.path.join(local_path, filename)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.threshold = threshold
        self.fill_holes = fill_holes
        self.close_gaps_kernel = close_gaps_kernel
        self.padding_iter = padding_iter
        self.tta_hflip = tta_hflip
        self.tta_vflip = tta_vflip

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type='scse'
        )

        convert_batchnorm_to_groupnorm(self.model.decoder)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def _run_inference(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32

        augmented = self.transform(image=image_rgb)
        tensor = augmented['image'].unsqueeze(0).to(self.device)

        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

        steps = 1 + int(self.tta_hflip) + int(self.tta_vflip)

        with torch.inference_mode():
            if self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = self.model(tensor)
                    probs = logits.sigmoid()
            else:
                logits = self.model(tensor)
                probs = logits.sigmoid()

            accumulated_probs = probs

            if self.tta_hflip:
                tensor_flip = torch.flip(tensor, [3])
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        probs_flip = self.model(tensor_flip).sigmoid()
                else:
                    probs_flip = self.model(tensor_flip).sigmoid()
                accumulated_probs += torch.flip(probs_flip, [3])

            if self.tta_vflip:
                tensor_flip = torch.flip(tensor, [2])
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        probs_flip = self.model(tensor_flip).sigmoid()
                else:
                    probs_flip = self.model(tensor_flip).sigmoid()
                accumulated_probs += torch.flip(probs_flip, [2])

        final_probs = accumulated_probs / steps
        prob_map = final_probs[0, 0, :h, :w].cpu().numpy()
        return prob_map

    def _postprocess(self, prob_map: np.ndarray) -> np.ndarray:
        binary_mask = (prob_map > self.threshold).astype(np.uint8) * 255

        if self.close_gaps_kernel > 0:
            k_size = int(self.close_gaps_kernel)
            if k_size % 2 == 0:
                k_size += 1
            kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_morph)

        if self.fill_holes:
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(binary_mask, contours, -1, 255, -1)

        if self.padding_iter > 0:
            kernel_pad = np.ones((3, 3), np.uint8)
            binary_mask = cv2.dilate(binary_mask, kernel_pad, iterations=int(self.padding_iter))

        return binary_mask

    def process(self, page: MangaPage) -> MangaPage:
        if page.original_image is None or not page.text_blocks:
            return page

        image_rgb = cv2.cvtColor(page.original_image, cv2.COLOR_BGR2RGB)
        full_prob_map = self._run_inference(image_rgb)
        full_mask = self._postprocess(full_prob_map)

        cropped_mask = np.zeros_like(full_mask)
        img_h, img_w = full_mask.shape

        for block in page.text_blocks:
            x1, y1, x2, y2 = block.bbox

            x1_c = max(0, min(x1, img_w))
            y1_c = max(0, min(y1, img_h))
            x2_c = max(0, min(x2, img_w))
            y2_c = max(0, min(y2, img_h))

            if x2_c > x1_c and y2_c > y1_c:
                cropped_mask[y1_c:y2_c, x1_c:x2_c] = full_mask[y1_c:y2_c, x1_c:x2_c]

        if page.mask is None or page.mask.shape[:2] != (img_h, img_w):
            page.mask = cropped_mask
        else:
            page.mask = cv2.bitwise_or(page.mask, cropped_mask)
        return page