import os
import json
from typing import List
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from src.modules.base import BaseModule
from src.core.context import MangaPage, TextBlock


class BaseTranslator(BaseModule):
    @staticmethod
    def _group_and_sort_blocks(blocks: List[TextBlock]) -> List[List[TextBlock]]:
        if not blocks:
            return []

        blocks.sort(key=lambda b: b.bbox[1])

        lines = []
        if blocks:
            lines.append([blocks[0]])
            for block in blocks[1:]:
                last_line = lines[-1]
                is_intersecting = any(max(b.bbox[1], block.bbox[1]) < min(b.bbox[3], block.bbox[3]) for b in last_line)
                if is_intersecting:
                    last_line.append(block)
                else:
                    lines.append([block])

        for line in lines:
            line.sort(key=lambda b: b.bbox[0], reverse=True)

        return lines

    def _translate(self, sorted_blocks: List[TextBlock]) -> str:
        raise NotImplementedError

    def process(self, page: MangaPage) -> MangaPage:
        if not page.text_blocks:
            return page

        all_blocks = page.text_blocks
        sorted_lines = self._group_and_sort_blocks(all_blocks)
        sorted_blocks_for_prompt = [block for line in sorted_lines for block in line]

        raw_response = self._translate(sorted_blocks_for_prompt)

        raw_response = raw_response.strip()
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3].strip()
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:].strip()
        elif raw_response.startswith("```"):
            raw_response = raw_response[3:].strip()

        raw_response = raw_response.replace('...', '…')

        translated_data = json.loads(raw_response)

        for i, block in enumerate(sorted_blocks_for_prompt):
            block.translated_text = translated_data[f'block_{i + 1}']

        return page


class SimpleTranslator(BaseTranslator):
    def process(self, page: MangaPage) -> MangaPage:
        for x in page.text_blocks:
            x.translated_text = x.source_text
        return page


class GemmaTranslator(BaseTranslator):
    def __init__(
            self,
            repo_id: str = "unsloth/gemma-4-E2B-it-GGUF",
            filename: str = "gemma-4-E2B-it-BF16.gguf",
            models_dir: str = "models",
            n_ctx: int = 2048,
            n_batch: int = 512,
            temperature: float = 0.0,
    ):
        from llama_cpp import Llama

        self.repo_id = repo_id
        self.filename = filename
        self.temperature = temperature
        self.local_path = self.get_model_dir(repo_id, models_dir)
        self.model_path = os.path.join(self.local_path, filename)
        self.marker_file = os.path.join(self.local_path, ".completed")

        if not self.is_downloaded():
            self._download_model()

        self.llm = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=-1,
            n_ctx=n_ctx,
            n_batch=n_batch,
            verbose=False,
        )

    @classmethod
    def get_model_dir(cls, repo_id: str, models_dir: str = "models") -> str:
        safe_folder_name = repo_id.replace("/", "_").replace("\\", "_")
        return os.path.join(models_dir, safe_folder_name)

    def is_downloaded(self) -> bool:
        return os.path.exists(self.model_path) and os.path.exists(self.marker_file)

    def _download_model(self):
        os.makedirs(self.local_path, exist_ok=True)
        hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
            local_dir=self.local_path,
            tqdm_class=tqdm
        )
        with open(self.marker_file, 'w') as f:
            f.write("completed")

    @classmethod
    def check_status(cls, repo_id: str, filename: str, models_dir: str = "models") -> tuple[bool, str]:
        local_path = cls.get_model_dir(repo_id, models_dir)
        model_file = os.path.join(local_path, filename)
        marker_file = os.path.join(local_path, ".completed")

        if os.path.exists(model_file) and os.path.exists(marker_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            return True, f"🟢 Модель загружена ({size_mb:.1f} MB)"
        elif os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            return False, f"🟡 Недокачана ({size_mb:.1f} MB)"
        else:
            return False, "🔴 Модель не найдена (скачается при старте)"

    def _translate(self, sorted_blocks) -> str:
        properties = {f'block_{i + 1}': {"type": "string"} for i in range(len(sorted_blocks))}
        sys_prompt = """Ты — профессиональный переводчик японской манги и комиксов.
        Переведи каждый блок в JSON на русский язык.
        """
        prompt = json.dumps({f'block_{i + 1}': block.source_text for i, block in enumerate(sorted_blocks)},
                            ensure_ascii=False)

        response = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties.keys()),
                    "additionalProperties": False
                }
            }
        )
        return response["choices"][0]["message"]["content"].strip()


class OpenAITranslator(BaseTranslator):
    def __init__(
            self,
            model_name: str = "gemini-3.5-flash-lite",
            api_key: str = None,
            base_url: str = None,
            temperature: float = 0.0,
    ):
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Не найден API Key.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.temperature = temperature

    def _translate(self, sorted_blocks) -> str:
        properties = {f'block_{i + 1}': {"type": "string"} for i in range(len(sorted_blocks))}
        sys_prompt = "Ты — профессиональный переводчик японской манги и комиксов. Переведи каждый блок в JSON на русский язык."
        prompt = json.dumps({f'block_{i + 1}': block.source_text for i, block in enumerate(sorted_blocks)},
                            ensure_ascii=False)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "manga_translation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": properties,
                        "required": list(properties.keys()),
                        "additionalProperties": False
                    }
                }
            }
        )
        return response.choices[0].message.content.strip()