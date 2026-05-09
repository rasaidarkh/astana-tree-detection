"""ModelAdapter — единый интерфейс для всех моделей детекции деревьев."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..schemas import Detection, ModelKind


class ModelAdapter(ABC):
    """Каждая модель оборачивается в adapter и кладёт результаты в общий формат `Detection`.

    Контракт минимальный: на вход путь к файлу, на выход список детекций
    в пиксельных координатах. GPS-привязку накладывает уже backend (geo.py).
    """

    kind: ModelKind
    name: str

    def __init__(self, **kwargs):
        self._loaded = False
        self._params = kwargs

    @abstractmethod
    def _load(self) -> None:
        """Lazy-загрузка весов модели. Вызывается один раз."""
        ...

    @abstractmethod
    def _predict_raw(self, image_path: str, confidence: float) -> list[Detection]:
        """Запустить инференс и вернуть детекции (без id, без GPS)."""
        ...

    def predict(self, image_path: str, confidence: float = 0.25) -> list[Detection]:
        """Public entry point: lazy-load + предсказание + проставление id."""
        if not self._loaded:
            self._load()
            self._loaded = True

        raw = self._predict_raw(image_path, confidence)
        for i, det in enumerate(raw):
            det.id = i + 1
        return raw

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "lazy"
        return f"<{self.__class__.__name__} kind={self.kind.value} {status}>"


class ModelRegistry:
    """Реестр инстансов адаптеров. Backend держит один экземпляр на каждую модель."""

    def __init__(self):
        self._adapters: dict[ModelKind, ModelAdapter] = {}

    def register(self, adapter: ModelAdapter) -> None:
        self._adapters[adapter.kind] = adapter

    def get(self, kind: ModelKind) -> Optional[ModelAdapter]:
        return self._adapters.get(kind)

    def available(self) -> list[ModelKind]:
        return list(self._adapters.keys())

    def __contains__(self, kind: ModelKind) -> bool:
        return kind in self._adapters
