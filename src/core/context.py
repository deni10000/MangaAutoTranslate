from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class Bubble:
    def __hash__(self):
        return id(self)

    """Model of a speech bubble"""
    bbox: Tuple[int, int, int, int]    # [x_min, y_min, x_max, y_max]
    polygon: Optional[List[Tuple[int, int]]] = None
    mask: Optional[np.ndarray] = None

@dataclass
class Panel:
    bbox: Tuple[int, int, int, int]    # [x_min, y_min, x_max, y_max]

@dataclass
class TextBlock:
    """Model of a text block on the page"""
    bbox: Tuple[int, int, int, int]    # [x_min, y_min, x_max, y_max]
    polygon: Optional[List[Tuple[int, int]]] = None
    source_text: str = ""
    translated_text: str = ""
    bubble: Optional[Bubble] = None
    font_size: Optional[int] = None
    font_path: Optional[str] = None
    text_color: Optional[str] = None
    stroke_color: Optional[str] = None
    stroke_width_ratio: Optional[float] = None
    rendering_box: Optional[Tuple[float, float, float, float, float]] = None  #(cx, cy, w, h, angle)

@dataclass
class MangaPage:
    """Complete state of the processed page"""
    image_path: str
    original_image: np.ndarray
    inpainted_image: Optional[np.ndarray] = None
    translated_image: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    pdf_data: Optional[bytes] = None
    
    bubbles: List[Bubble] = field(default_factory=list)
    text_blocks: List[TextBlock] = field(default_factory=list)
    panels: List[Panel] = field(default_factory=list)
