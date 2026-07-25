import os
import sys
import time
import json
import hashlib
import glob
import streamlit as st

CONFIG_FILE = "translator_config.json"

DEFAULT_CONFIG = {
    "translator_type": "Gemma (Локальный)",
    "gemma_repo_id": "unsloth/gemma-4-E2B-it-GGUF",
    "gemma_filename": "gemma-4-E2B-it-BF16.gguf",
    "openai_model_name": "gemini-3.5-flash-lite",
    "openai_api_key": "",
    "openai_base_url": "",
    "selected_fonts": ["MP Manga.ttf", "NotoSans-BoldItalic.ttf"],
    "ocr_language": "Японский (MangaOcr)",
}

OCR_LANGUAGES = {
    "Японский (MangaOcr)": ("manga", None),
    "Корейский (PaddleOCR)": ("paddle", "korean"),
    "Китайский (PaddleOCR)": ("paddle", "ch"),
    "Английский (PaddleOCR)": ("paddle", "en"),
}


def get_available_fonts() -> list:
    font_extensions = ("*.ttf", "*.otf", "*.woff", "*.woff2")
    search_paths = [".", "fonts", "assets", "assets/fonts"]
    found_fonts = []

    for path in search_paths:
        if os.path.exists(path):
            for ext in font_extensions:
                for font_path in glob.glob(os.path.join(path, ext)):
                    norm_path = os.path.normpath(font_path)
                    if norm_path not in found_fonts:
                        found_fonts.append(norm_path)

    for df in DEFAULT_CONFIG["selected_fonts"]:
        if df not in found_fonts:
            found_fonts.append(df)

    return sorted(found_fonts)


def load_translator_config() -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                cfg.update(saved)
        except Exception:
            pass

    if not cfg.get("gemma_repo_id") or not str(cfg.get("gemma_repo_id")).strip():
        cfg["gemma_repo_id"] = DEFAULT_CONFIG["gemma_repo_id"]
    if not cfg.get("gemma_filename") or not str(cfg.get("gemma_filename")).strip():
        cfg["gemma_filename"] = DEFAULT_CONFIG["gemma_filename"]
    if not cfg.get("selected_fonts") or not isinstance(cfg.get("selected_fonts"), list):
        cfg["selected_fonts"] = DEFAULT_CONFIG["selected_fonts"].copy()
    if cfg.get("ocr_language") not in OCR_LANGUAGES:
        cfg["ocr_language"] = DEFAULT_CONFIG["ocr_language"]

    return cfg


def save_translator_config():
    if "app_config" not in st.session_state:
        return

    config = st.session_state.app_config

    if not config.get("gemma_repo_id") or not str(config["gemma_repo_id"]).strip():
        config["gemma_repo_id"] = DEFAULT_CONFIG["gemma_repo_id"]
    if not config.get("gemma_filename") or not str(config["gemma_filename"]).strip():
        config["gemma_filename"] = DEFAULT_CONFIG["gemma_filename"]
    if not config.get("selected_fonts"):
        config["selected_fonts"] = DEFAULT_CONFIG["selected_fonts"].copy()
    if config.get("ocr_language") not in OCR_LANGUAGES:
        config["ocr_language"] = DEFAULT_CONFIG["ocr_language"]

    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
    except Exception:
        pass


def init_session_config():
    if "app_config" not in st.session_state:
        st.session_state.app_config = load_translator_config()

    if "show_translated_text" not in st.session_state:
        st.session_state.show_translated_text = False


def enable_translated_text():
    st.session_state.show_translated_text = True


def clear_text_and_font_keys():
    keys_to_remove = [
        k for k in st.session_state.keys()
        if k.startswith("text_edit_") or k.startswith("font_size_")
    ]
    for k in keys_to_remove:
        del st.session_state[k]


def update_config_field(field_name):
    widget_key = f"input_{field_name}"
    if widget_key in st.session_state:
        st.session_state.app_config[field_name] = st.session_state[widget_key]
        save_translator_config()


def update_translator_type():
    if "input_translator_type" in st.session_state:
        st.session_state.app_config["translator_type"] = st.session_state["input_translator_type"]
        save_translator_config()


@st.cache_resource(show_spinner=False)
def get_cached_gemma_translator(repo_id: str, filename: str):
    from src.modules.translation.translator import GemmaTranslator
    return GemmaTranslator(repo_id=repo_id, filename=filename)


@st.cache_resource(show_spinner=False)
def get_cached_openai_translator(model_name: str, api_key: str, base_url: str):
    from src.modules.translation.translator import OpenAITranslator
    return OpenAITranslator(
        model_name=model_name,
        api_key=api_key or None,
        base_url=base_url or None
    )


@st.cache_resource(show_spinner=False)
def get_cached_simple_translator():
    from src.modules.translation.translator import SimpleTranslator
    return SimpleTranslator()


def get_translator_instance():
    cfg = st.session_state.app_config
    t_type = cfg.get("translator_type", DEFAULT_CONFIG["translator_type"])

    if t_type == "Gemma (Локальный)":
        repo_id = cfg.get("gemma_repo_id") or DEFAULT_CONFIG["gemma_repo_id"]
        filename = cfg.get("gemma_filename") or DEFAULT_CONFIG["gemma_filename"]
        return get_cached_gemma_translator(repo_id=repo_id, filename=filename)

    elif t_type == "OpenAI / API":
        return get_cached_openai_translator(
            model_name=cfg.get("openai_model_name", DEFAULT_CONFIG["openai_model_name"]),
            api_key=cfg.get("openai_api_key", ""),
            base_url=cfg.get("openai_base_url", "")
        )
    else:
        return get_cached_simple_translator()


@st.cache_resource(show_spinner=False)
def get_cached_manga_ocr():
    from src.modules.ocr.ocr import MangaOcr
    return MangaOcr()


@st.cache_resource(show_spinner=False)
def get_cached_paddle_ocr(lang: str):
    from src.modules.ocr.ocr import PaddleOcr
    return PaddleOcr(lang=lang)


@st.cache_resource(show_spinner=False)
def get_cached_easy_ocr(lang: str):
    from src.modules.ocr.ocr import EasyOcr
    return EasyOcr(lang=lang)


def get_ocr_instance(ocr_lang_label: str):
    engine, lang_code = OCR_LANGUAGES.get(ocr_lang_label, ("manga", None))
    if engine == "paddle":
        return get_cached_paddle_ocr(lang_code)
    if engine == "easyocr":
        return get_cached_easy_ocr(lang_code)
    return get_cached_manga_ocr()


@st.cache_resource(show_spinner=False)
def get_cached_pdf_renderer(fonts_key: str):
    from src.modules.rendering.PDF_renderer import PDFRenderer
    fonts = fonts_key.split("||")
    return PDFRenderer(default_font_path=fonts)


def get_pdf_renderer_instance():
    cfg = st.session_state.app_config
    fonts = cfg.get("selected_fonts") or DEFAULT_CONFIG["selected_fonts"]
    fonts_key = "||".join(fonts)
    return get_cached_pdf_renderer(fonts_key)


def get_module_device(module) -> str:
    if module is None:
        return "Не загружена"
    if hasattr(module, 'llm'):
        return "GPU / CPU (llama.cpp)"
    if hasattr(module, 'client'):
        return "Cloud API"
    if module.__class__.__name__ == "SimpleTranslator":
        return "Без ИИ"
    if module.__class__.__name__ == "PaddleOcr":
        return "PaddleOCR"
    if module.__class__.__name__ == "EasyOcr":
        return "EasyOCR"
    if module.__class__.__name__ == "MangaOcr":
        return str(getattr(module, 'device', 'GPU / CPU'))
    if hasattr(module, 'device'):
        return str(module.device)
    if hasattr(module, 'model'):
        return get_module_device(module.model)
    return "CPU / Unknown"


def get_image_hash(img) -> str:
    return hashlib.md5(img.tobytes()).hexdigest()[:12]


@st.cache_resource
def load_pipeline_modules():
    from src.modules.detection.yolo_detector import YoloV26SegmentationDetector
    from src.modules.masking.smp_masker import SmpTextMasker
    from src.modules.filter.empty_elements_filter import EmptyElementsFilter
    from src.modules.inpainting.fast_bubble_cleaner import FastBubbleCleaner
    from src.modules.inpainting.lama_inpainter import LamaInpainter
    from src.modules.filter.text_rendering_box_assigner import TextRenderingBoxAssigner

    yolo = YoloV26SegmentationDetector()
    masker = SmpTextMasker()
    empty_filter = EmptyElementsFilter()
    fast_cleaner = FastBubbleCleaner()
    lama = LamaInpainter()
    assigner = TextRenderingBoxAssigner()

    return yolo, masker, empty_filter, fast_cleaner, lama, assigner


def apply_pipeline_params(yolo, masker, fast_cleaner, assigner, pdf_renderer, params: dict):
    yolo.conf_threshold = params["conf_threshold"]
    masker.threshold = params["smp_threshold"]
    fast_cleaner.spread_threshold = params["spread_threshold"]
    assigner.bubble_padding_ratio = params["bubble_padding_ratio"]
    pdf_renderer.stroke_width_ratio = params["stroke_width_ratio"]


def get_distinct_colors(n: int) -> list:
    import cv2
    import numpy as np
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        color_hsv = np.uint8([[[hue, 220, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
        colors.append(color_bgr)
    return colors


def draw_rotated_box(img, rendering_box: tuple, color=(255, 0, 255), thickness=2):
    import cv2
    import numpy as np
    if rendering_box is None:
        return
    cx, cy, w, h, angle = rendering_box
    rect = ((float(cx), float(cy)), (float(w), float(h)), float(angle))
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    cv2.polylines(img, [box], isClosed=True, color=color, thickness=thickness)
    cv2.circle(img, (int(cx), int(cy)), 3, color, -1)


def create_visualization(
        page,
        mask_yolo=None,
        mask_smp=None,
        mask_constrained=None,
        show_translated_bg: bool = False,
        show_inpainted: bool = True,
        show_bubbles: bool = False,
        show_bubble_masks: bool = False,
        show_text_bbox: bool = False,
        show_mask_yolo: bool = False,
        show_mask_smp: bool = False,
        show_mask_constrained: bool = False,
        show_rendering_boxes: bool = False,
        mask_alpha: float = 0.5
):
    import cv2
    import numpy as np

    if show_translated_bg and page.translated_image is not None:
        vis_img = page.translated_image.copy()
    elif show_inpainted and page.inpainted_image is not None:
        vis_img = page.inpainted_image.copy()
    else:
        vis_img = page.original_image.copy()

    def apply_alpha_overlay(base_img, overlay_img, mask, alpha):
        if np.any(mask):
            base_img[mask] = (
                    base_img[mask].astype(np.float32) * (1.0 - alpha) +
                    overlay_img[mask].astype(np.float32) * alpha
            ).astype(np.uint8)

    if show_bubble_masks and page.bubbles:
        bubble_overlay = np.zeros_like(vis_img)
        colors = get_distinct_colors(len(page.bubbles))
        for idx, b in enumerate(page.bubbles):
            if b.mask is not None and b.mask.size > 0 and b.bbox is not None:
                x1, y1, x2, y2 = map(int, b.bbox)
                h_crop, w_crop = b.mask.shape[:2]
                color = colors[idx % len(colors)]
                m_bool = b.mask > 0
                sub_overlay = bubble_overlay[y1:y1 + h_crop, x1:x1 + w_crop]
                sub_overlay[m_bool] = color
        m_total = np.any(bubble_overlay > 0, axis=2)
        apply_alpha_overlay(vis_img, bubble_overlay, m_total, mask_alpha)

    if show_mask_yolo and mask_yolo is not None and np.any(mask_yolo):
        blue_overlay = np.zeros_like(vis_img)
        blue_overlay[:, :] = (255, 128, 0)
        apply_alpha_overlay(vis_img, blue_overlay, mask_yolo > 0, mask_alpha)

    if show_mask_smp and mask_smp is not None and np.any(mask_smp):
        red_overlay = np.zeros_like(vis_img)
        red_overlay[:, :] = (0, 0, 255)
        apply_alpha_overlay(vis_img, red_overlay, mask_smp > 0, mask_alpha)

    if show_mask_constrained and mask_constrained is not None and np.any(mask_constrained):
        green_overlay = np.zeros_like(vis_img)
        green_overlay[:, :] = (0, 255, 0)
        apply_alpha_overlay(vis_img, green_overlay, mask_constrained > 0, mask_alpha)

    if show_bubbles:
        for b in page.bubbles:
            if b.bbox is not None:
                x1, y1, x2, y2 = map(int, b.bbox)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

    if show_text_bbox:
        for tb in page.text_blocks:
            if tb.bbox is not None:
                x1, y1, x2, y2 = map(int, tb.bbox)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 255, 0), 2)

    if show_rendering_boxes:
        for tb in page.text_blocks:
            if tb.rendering_box is not None:
                draw_rotated_box(vis_img, tb.rendering_box, color=(255, 0, 255), thickness=2)

    return cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)


def run_app():
    st.set_page_config(layout="wide", page_title="Manga Pipeline Visualizer")
    st.title("🖼️ Manga Pipeline Debugger & Visualizer")

    init_session_config()

    with st.spinner("Загрузка базовых нейросетей..."):
        yolo, masker, empty_filter, fast_cleaner, lama, assigner = load_pipeline_modules()

    st.sidebar.title("⚙️ Панель управления")

    with st.sidebar.expander("📁 Загрузка файла", expanded=True):
        uploaded_file = st.file_uploader("Выберите манга-страницу", type=["jpg", "png", "jpeg", "webp"])

    st.sidebar.markdown("**🌐 Язык распознавания (OCR):**")
    ocr_options = list(OCR_LANGUAGES.keys())
    cfg_ocr = st.session_state.app_config
    current_ocr = cfg_ocr.get("ocr_language", DEFAULT_CONFIG["ocr_language"])
    ocr_default_index = ocr_options.index(current_ocr) if current_ocr in ocr_options else 0

    ocr_lang_label = st.sidebar.selectbox(
        "Язык распознавания",
        ocr_options,
        index=ocr_default_index,
        key="input_ocr_language",
        on_change=update_config_field,
        args=("ocr_language",),
        label_visibility="collapsed",
        help="Японский — MangaOcr, корейский/китайский/английский — PaddleOCR. Модель загружается при первом запуске распознавания."
    )

    with st.sidebar.expander("🔧 Параметры пайплайна", expanded=False):
        st.markdown("**Детекция (YOLO):**")
        conf_threshold = st.slider(
            "Порог уверенности (conf)",
            min_value=0.01, max_value=0.5, value=0.05, step=0.01,
            help="Ниже — больше детекций, но больше шума"
        )

        st.markdown("**Маскирование (SMP):**")
        smp_threshold = st.slider(
            "Порог бинаризации маски",
            min_value=0.005, max_value=0.2, value=0.01, step=0.005,
            help="Ниже — более полная маска текста"
        )

        st.markdown("**Очистка пузырей:**")
        spread_threshold = st.slider(
            "Порог разброса (заливка)",
            min_value=5.0, max_value=80.0, value=20.0, step=1.0,
            help="Ниже — агрессивнее заливка однородных областей"
        )

        st.markdown("**Ограничение маски:**")
        shrink_margin = st.slider(
            "Отступ сужения бабла",
            min_value=0, max_value=15, value=5,
            help="На сколько пикселей сузить маску пузыря от краёв"
        )

        st.markdown("**Рендеринг:**")
        bubble_padding_ratio = st.slider(
            "Отступ текста от краёв пузыря",
            min_value=0.0, max_value=0.2, value=0.05, step=0.01,
            help="Доля размера пузыря, вычитаемая из rendering box"
        )
        stroke_width_ratio = st.slider(
            "Толщина обводки текста",
            min_value=0.0, max_value=0.15, value=0.06, step=0.005,
            help="Относительно размера шрифта"
        )

    with st.sidebar.expander("🔤 Настройки шрифтов", expanded=False):
        available_fonts = get_available_fonts()
        saved_fonts = st.session_state.app_config.get("selected_fonts", DEFAULT_CONFIG["selected_fonts"])

        valid_defaults = [f for f in saved_fonts if f in available_fonts]
        if not valid_defaults and available_fonts:
            valid_defaults = [available_fonts[0]]

        selected_fonts = st.multiselect(
            "Приоритет шрифтов (основной + fallback):",
            options=available_fonts,
            default=valid_defaults,
            key="input_selected_fonts",
            on_change=update_config_field,
            args=("selected_fonts",),
            help="Порядок выбора определяет приоритет. Первый шрифт — основной, последующие используются для отсутствующих символов."
        )

        if selected_fonts:
            st.caption(f"**Основной шрифт:** `{selected_fonts[0]}`")
            if len(selected_fonts) > 1:
                st.caption(f"**Fallback:** `{', '.join(selected_fonts[1:])}`")

    with st.sidebar.expander("🎨 Переключатели слоев и масок", expanded=False):
        show_inpainted = st.checkbox("🖼️ Показать закрашенный фон (Inpainted)", value=True)
        show_translated_text = st.checkbox(
            "🌐 Показать переведённый текст на изображении",
            key="show_translated_text"
        )

        st.markdown("---")
        show_bubbles = st.checkbox("BBox Баблов (Белый)", value=False)
        show_bubble_masks = st.checkbox("🎨 Разноцветные маски баблов", value=False)
        show_text_bbox = st.checkbox("BBox Текста (Голубой)", value=False)

        st.markdown("---")
        st.markdown("**Маски текста:**")
        show_mask_yolo = st.checkbox("Сырая маска YOLO (Синий)", value=False)
        show_mask_smp = st.checkbox("Сырая маска SMP (Красный)", value=False)
        show_mask_constrained = st.checkbox("Итоговая ограниченная маска (Зеленый)", value=False)
        show_rendering_boxes = st.checkbox("Rendering Box (Фиолетовый)", value=False)
        mask_alpha = st.slider("Прозрачность масок", 0.0, 1.0, 0.4)

    with st.sidebar.expander("🖥️ Устройства моделей", expanded=False):
        st.write(f"**YOLO Detector:** `{get_module_device(yolo)}`")
        st.write(f"**SMP Masker:** `{get_module_device(masker)}`")
        ocr_obj = st.session_state.get("active_ocr_module", None)
        st.write(f"**OCR ({ocr_lang_label}):** `{get_module_device(ocr_obj)}`")
        st.write(f"**Lama Inpainter:** `{get_module_device(lama)}`")
        translator_obj = st.session_state.get("active_translator_module", None)
        st.write(f"**Translator:** `{get_module_device(translator_obj)}`")

    with st.sidebar.expander("🌐 Настройки переводчика", expanded=False):
        translator_options = ["Gemma (Локальный)", "OpenAI / API", "Без перевода (Simple)"]

        cfg = st.session_state.app_config
        current_type = cfg.get("translator_type", DEFAULT_CONFIG["translator_type"])
        default_index = translator_options.index(current_type) if current_type in translator_options else 0

        selected_type = st.selectbox(
            "Тип перевода",
            translator_options,
            index=default_index,
            key="input_translator_type",
            on_change=update_translator_type
        )

        if selected_type == "Gemma (Локальный)":
            st.text_input(
                "Repo ID",
                value=cfg.get("gemma_repo_id", DEFAULT_CONFIG["gemma_repo_id"]),
                key="input_gemma_repo_id",
                on_change=update_config_field,
                args=("gemma_repo_id",)
            )
            st.text_input(
                "GGUF Filename",
                value=cfg.get("gemma_filename", DEFAULT_CONFIG["gemma_filename"]),
                key="input_gemma_filename",
                on_change=update_config_field,
                args=("gemma_filename",)
            )

            from src.modules.translation.translator import GemmaTranslator
            is_ready, status_msg = GemmaTranslator.check_status(
                repo_id=cfg.get("gemma_repo_id", DEFAULT_CONFIG["gemma_repo_id"]),
                filename=cfg.get("gemma_filename", DEFAULT_CONFIG["gemma_filename"])
            )
            st.caption(f"**Статус модели:** {status_msg}")

        elif selected_type == "OpenAI / API":
            st.text_input(
                "Model Name",
                value=cfg.get("openai_model_name", DEFAULT_CONFIG["openai_model_name"]),
                key="input_openai_model_name",
                on_change=update_config_field,
                args=("openai_model_name",)
            )
            st.text_input(
                "API Key",
                value=cfg.get("openai_api_key", DEFAULT_CONFIG["openai_api_key"]),
                type="password",
                key="input_openai_api_key",
                on_change=update_config_field,
                args=("openai_api_key",)
            )
            st.text_input(
                "Base URL (Опционально)",
                value=cfg.get("openai_base_url", DEFAULT_CONFIG["openai_base_url"]),
                key="input_openai_base_url",
                on_change=update_config_field,
                args=("openai_base_url",)
            )

    st.sidebar.markdown("---")
    translate_clicked = st.sidebar.button(
        "🌐 Перевести текст",
        width="stretch",
        on_click=enable_translated_text
    )
    rerender_clicked = st.sidebar.button(
        "🔄 Перерисовать с правками",
        width="stretch",
        on_click=enable_translated_text
    )

    if uploaded_file is not None:
        import cv2
        import numpy as np
        from src.core.context import MangaPage
        from src.modules.filter.text_mask_constrainer import TextMaskConstrainer

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img_hash = get_image_hash(img_bgr)

        if st.session_state.get("current_img_hash") != img_hash:
            st.session_state.current_img_hash = img_hash
            clear_text_and_font_keys()

        pdf_renderer = get_pdf_renderer_instance()

        pipeline_params = {
            "conf_threshold": conf_threshold,
            "smp_threshold": smp_threshold,
            "spread_threshold": spread_threshold,
            "bubble_padding_ratio": bubble_padding_ratio,
            "stroke_width_ratio": stroke_width_ratio,
        }
        apply_pipeline_params(yolo, masker, fast_cleaner, assigner, pdf_renderer, pipeline_params)

        state_key = f"{uploaded_file.name}_{img_hash}_{ocr_lang_label}_{shrink_margin}_{conf_threshold}_{smp_threshold}_{spread_threshold}_{bubble_padding_ratio}"
        reprocess_needed = ("cache_key" not in st.session_state) or (st.session_state.cache_key != state_key)

        if reprocess_needed or st.sidebar.button("♻️ Перезапустить пайплайн", width="stretch"):
            st.session_state.pop("cached_page", None)
            st.session_state.pop("cached_yolo", None)
            st.session_state.pop("cached_smp", None)
            st.session_state.pop("cached_constrained", None)
            st.session_state.pop("timing_info", None)
            clear_text_and_font_keys()

            timings = {}
            pipeline_start = time.time()
            status_box = st.status("⏳ Выполнение базового пайплайна...", expanded=True)

            def run_step(step_name, func):
                t_start = time.time()
                status_box.update(
                    label=f"⏳ [{step_name}] Выполняется... (Прошло: {time.time() - pipeline_start:.1f} с)")
                res = func()
                t_dur = time.time() - t_start
                timings[step_name] = t_dur
                status_box.write(f"✅ **{step_name}**: {t_dur:.2f} с")
                return res

            page = MangaPage(image_path=uploaded_file.name, original_image=img_bgr)

            page = run_step("YOLO Detection", lambda: yolo.process(page))
            yolo_mask = page.mask.copy() if page.mask is not None else None

            page.mask = None
            page = run_step("SMP Masking", lambda: masker.process(page))
            smp_mask = page.mask.copy() if page.mask is not None else None

            if yolo_mask is not None and smp_mask is not None:
                page.mask = cv2.bitwise_or(yolo_mask, smp_mask)
            elif smp_mask is not None:
                page.mask = smp_mask
            elif yolo_mask is not None:
                page.mask = yolo_mask

            page = run_step("Mask Constrainer",
                            lambda: TextMaskConstrainer(max_bubble_shrink_margin=shrink_margin).process(page))
            constrained_mask = page.mask.copy() if page.mask is not None else None

            def do_ocr():
                ocr_instance = get_ocr_instance(ocr_lang_label)
                st.session_state.active_ocr_module = ocr_instance
                return ocr_instance.process(page)

            page = run_step("OCR", do_ocr)
            page = run_step("Empty Filter", lambda: empty_filter.process(page))
            page = run_step("Fast Bubble Cleaner", lambda: fast_cleaner.process(page))
            page = run_step("LaMa Inpainting", lambda: lama.process(page))
            page = run_step("Rendering Box Assigner", lambda: assigner.process(page))

            timings["Total Pipeline"] = sum(timings.values())
            status_box.update(label=f"🎉 Завершено за {timings['Total Pipeline']:.2f} с!", state="complete",
                              expanded=False)

            st.session_state.cached_page = page
            st.session_state.cached_yolo = yolo_mask
            st.session_state.cached_smp = smp_mask
            st.session_state.cached_constrained = constrained_mask
            st.session_state.cache_key = state_key
            st.session_state.timing_info = timings

        page = st.session_state.cached_page
        yolo_mask = st.session_state.cached_yolo
        smp_mask = st.session_state.cached_smp
        constrained_mask = st.session_state.cached_constrained
        timings = st.session_state.get("timing_info", {})

        if translate_clicked:
            status_box = st.status("🌐 Выполнение перевода и рендеринга...", expanded=True)
            trans_start = time.time()

            save_translator_config()

            with st.spinner("Получение экземпляра переводчика из кэша..."):
                translator = get_translator_instance()
                st.session_state.active_translator_module = translator

            t0 = time.time()
            status_box.update(label=f"⏳ [Translation] Переводим текст... (Прошло: {time.time() - trans_start:.1f} с)")
            page = translator.process(page)
            timings["Translation"] = time.time() - t0
            status_box.write(f"✅ **Translation**: {timings['Translation']:.2f} с")

            for tb in page.text_blocks:
                tb.font_size = None

            t0 = time.time()
            status_box.update(label=f"⏳ [PDF Rendering] Генерация PDF... (Прошло: {time.time() - trans_start:.1f} с)")
            page = pdf_renderer.process(page)
            timings["PDF Rendering"] = time.time() - t0
            status_box.write(f"✅ **PDF Rendering**: {timings['PDF Rendering']:.2f} с")

            for i, tb in enumerate(page.text_blocks):
                widget_key = f"text_edit_{img_hash}_{i}"
                font_key = f"font_size_{img_hash}_{i}"
                if tb.translated_text:
                    st.session_state[widget_key] = tb.translated_text
                if getattr(tb, 'font_size', None) is not None:
                    st.session_state[font_key] = int(tb.font_size)

            status_box.update(label="🎉 Перевод и рендеринг завершены!", state="complete", expanded=False)
            st.session_state.cached_page = page
            st.session_state.timing_info = timings

        if rerender_clicked:
            for i, block in enumerate(page.text_blocks):
                widget_key = f"text_edit_{img_hash}_{i}"
                font_key = f"font_size_{img_hash}_{i}"
                if widget_key in st.session_state:
                    block.translated_text = st.session_state[widget_key]
                if font_key in st.session_state:
                    block.font_size = st.session_state[font_key]

            status_box = st.status("🔄 Перерисовка PDF...", expanded=True)
            t0 = time.time()
            page = pdf_renderer.process(page)
            timings["PDF Re-render"] = time.time() - t0
            status_box.write(f"✅ **PDF Re-render**: {timings['PDF Re-render']:.2f} с")
            status_box.update(label="🎉 Перерисовка завершена!", state="complete", expanded=False)

            st.session_state.cached_page = page
            st.session_state.timing_info = timings

        vis_result = create_visualization(
            page=page,
            mask_yolo=yolo_mask,
            mask_smp=smp_mask,
            mask_constrained=constrained_mask,
            show_translated_bg=st.session_state.get("show_translated_text", False),
            show_inpainted=show_inpainted,
            show_bubbles=show_bubbles,
            show_bubble_masks=show_bubble_masks,
            show_text_bbox=show_text_bbox,
            show_mask_yolo=show_mask_yolo,
            show_mask_smp=show_mask_smp,
            show_mask_constrained=show_mask_constrained,
            show_rendering_boxes=show_rendering_boxes,
            mask_alpha=mask_alpha
        )

        col1, col2 = st.columns([2.5, 1])

        with col1:
            st.subheader("Визуализация слоев")
            st.image(vis_result, width="stretch")

            if page.pdf_data is not None:
                st.download_button(
                    label="📥 Скачать результат в PDF",
                    data=page.pdf_data,
                    file_name=f"translated_{os.path.splitext(uploaded_file.name)[0]}.pdf",
                    mime="application/pdf"
                )

            if timings:
                st.markdown("---")
                st.subheader("⏱️ Время обработки")
                timing_cols = st.columns(3)
                for idx, (name, duration) in enumerate(timings.items()):
                    with timing_cols[idx % 3]:
                        st.metric(label=name, value=f"{duration:.2f} с")

        with col2:
            st.subheader("📊 Информация об объектах")
            st.write(f"**Найдено баблов:** {len(page.bubbles)}")
            st.write(f"**Найдено блоков текста:** {len(page.text_blocks)}")

            st.markdown("---")
            st.subheader("📝 Блоки редактирования текста")

            if not page.text_blocks:
                st.info("Текстовые блоки не найдены.")
            else:
                for i, tb in enumerate(page.text_blocks):
                    widget_key = f"text_edit_{img_hash}_{i}"
                    font_key = f"font_size_{img_hash}_{i}"

                    with st.expander(f"Блок {i + 1}", expanded=False):
                        st.markdown("**Оригинал (OCR):**")
                        st.code(tb.source_text if tb.source_text else "[Пусто]", language=None)

                        st.markdown("**Перевод (редактируемый):**")

                        if widget_key not in st.session_state:
                            st.session_state[widget_key] = tb.translated_text if tb.translated_text else ""

                        st.text_area(
                            label=f"Перевод блока {i + 1}",
                            key=widget_key,
                            label_visibility="collapsed",
                            height=80
                        )

                        if font_key not in st.session_state:
                            st.session_state[font_key] = int(getattr(tb, 'font_size', None) or 16)

                        st.number_input(
                            label="Размер шрифта (px)",
                            min_value=4,
                            max_value=100,
                            step=1,
                            key=font_key
                        )

                        if tb.bubble is not None:
                            st.caption(f"🔗 В бабле: `{tb.bubble.bbox}`")
                        else:
                            st.caption("📌 Свободный текст")

                        if tb.rendering_box is not None:
                            cx, cy, w, h, angle = tb.rendering_box
                            st.caption(f"📐 Box: {w:.0f}×{h:.0f}, {angle:.1f}°")

    else:
        st.info("👆 Загрузите изображение манга-страницы.")


if __name__ == "__main__":
    if st.runtime.exists():
        run_app()
    else:
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())