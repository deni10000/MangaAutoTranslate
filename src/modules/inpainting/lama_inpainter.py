import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from huggingface_hub import hf_hub_download

from src.modules.base import BaseModule
from src.core.context import MangaPage


class LamaInpainter(BaseModule):
    def __init__(
            self,
            repo_id: str = "df1412/anime-big-lama",
            filename: str = "anime-manga-big-lama.pt",
            device: Optional[str] = None,
            models_dir: str = "models",
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        local_path = os.path.join(models_dir, "lama_inpainter_model")
        marker_file = os.path.join(local_path, ".completed")

        if not (os.path.exists(local_path) and os.path.exists(marker_file)):
            os.makedirs(local_path, exist_ok=True)
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_path)
            with open(marker_file, 'w') as f:
                f.write("completed")

        model_path = os.path.join(local_path, filename)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def process(self, page: MangaPage) -> MangaPage:
        if page.inpainted_image is not None:
            base_img = page.inpainted_image
        else:
            base_img = page.original_image.copy()

        if page.mask is None or not np.any(page.mask):
            page.inpainted_image = base_img
            return page

        h, w = base_img.shape[:2]
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8

        img = base_img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device, dtype=torch.float32)
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        mask_binary = (page.mask > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
        if pad_h > 0 or pad_w > 0:
            mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

        with torch.inference_mode():
            output = self.model(img_tensor, mask_tensor)

        output = output[:, :, :h, :w]
        output = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        output = (np.clip(output, 0, 1) * 255).astype(np.uint8)

        mask_3ch = (page.mask > 0)[..., None]
        page.inpainted_image = np.where(mask_3ch, output, base_img)

        return page