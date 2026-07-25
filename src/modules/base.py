from src.core.context import MangaPage

class BaseModule:
    def process(self, page: MangaPage) -> MangaPage:
        raise NotImplementedError
