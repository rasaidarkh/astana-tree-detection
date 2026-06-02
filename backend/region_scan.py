"""Auto-Zoom Region Scan — Layer 0 of the tiled-inference stack.

Проблема: пользователь выбирает большой прямоугольник на Leaflet (километр+),
зум на момент рисования любой. Скачивать ОДИН большой снимок на zoom=19 —
не получится: `capture_bbox` рубит >144 тайлов (защита MAX_TILES). А скачать
на маленьком zoom (например 16) — это GSD ~2.4 м/px, кроны деревьев = 4–6
пикселей, модель такое не видит.

Решение: фиксируем zoom = 19 (max Esri-зум, ~0.3 м/px), а большой bbox
автоматически дробим в NxM-сетку под-регионов, каждый ≤ MAX_TILES_PER_SUB
тайлов. На каждом под-регионе запускаем тот же pipeline (capture+predict),
результаты собираются в БД и появляются в city-aggregate map.

Это Layer 0 в иерархии:
  - Layer 0 (этот модуль): bbox → список под-bbox-ов фиксированного зума
  - Layer 1 (`map_capture.capture_bbox`): bbox+zoom → склеенная картинка из
    256×256 ESRI-тайлов, обрезанная точно по bbox
  - Layer 2 (адаптер `_predict_tiled`): большая картинка → 640×640 окна,
    global NMS поверх детекций

Все три уровня независимы: пользователь может скрутить только Layer 2
(одиночный capture, тайлы внутри адаптера), или только Layer 0+1
(сетка + capture без тайлов в адаптере), или всё сразу.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .map_capture import MAX_TILES, TILE_PROVIDERS, latlng_to_tile_xy

# Запас от MAX_TILES — оставляем место для краёв ceil(...) при дроблении
# и не упираемся в потолок ровно. capture_bbox внутри посмотрит ещё раз.
DEFAULT_MAX_TILES_PER_SUB = 100

# Лимит на число под-регионов в одном запросе — каждый под-регион тащит
# сотни тайлов + прогоняет модель. Поднят до 64, чтобы покрыть ~1 км² на
# zoom=20 (Google, ~0.15 м/px): на z20 та же площадь даёт в 4× больше тайлов,
# чем на z19, поэтому 1×1 км рассыпается на ~36 под-регионов. Сканы стали
# медленнее (минуты), но это сознательный размен на детализацию.
DEFAULT_MAX_SUBREGIONS = 64


@dataclass
class SubRegion:
    """Один под-прямоугольник в сетке. row/col — для отладки и UI-прогресса."""
    row: int
    col: int
    nw_lat: float
    nw_lng: float
    se_lat: float
    se_lng: float


def count_tiles_for_bbox(
    nw_lat: float, nw_lng: float, se_lat: float, se_lng: float, zoom: int
) -> int:
    """Сколько 256×256 ESRI-тайлов покрывает bbox на заданном zoom.

    Совпадает с тем, что `capture_bbox` посчитает внутри — используем одну
    и ту же формулу `latlng_to_tile_xy`, чтобы Layer 0 и Layer 1 не разъехались.
    """
    nx_f, ny_f = latlng_to_tile_xy(nw_lat, nw_lng, zoom)
    sx_f, sy_f = latlng_to_tile_xy(se_lat, se_lng, zoom)
    x_min, x_max = int(math.floor(nx_f)), int(math.floor(sx_f))
    y_min, y_max = int(math.floor(ny_f)), int(math.floor(sy_f))
    cols = x_max - x_min + 1
    rows = y_max - y_min + 1
    return cols * rows


def split_bbox_to_subregions(
    nw_lat: float,
    nw_lng: float,
    se_lat: float,
    se_lng: float,
    zoom: int,
    max_tiles_per_sub: int = DEFAULT_MAX_TILES_PER_SUB,
) -> list[SubRegion]:
    """Дробим bbox на сетку NxM под-регионов так, чтобы каждый ≤ max_tiles_per_sub.

    Алгоритм:
      1. Считаем сколько тайлов нужно на ВСЁ bbox на заданном zoom.
      2. Если влезает в `max_tiles_per_sub` — возвращаем один SubRegion = весь bbox.
      3. Иначе считаем сторону квадратной сетки: `n = ceil(sqrt(total / max))`,
         так что каждая ячейка получает ≤ max_tiles_per_sub тайлов.
      4. Возвращаем ровно n×n под-bbox-ов, равномерно разделённых по lat/lng.

    Делим по lat/lng линейно — это даёт неидеально-равные тайлы в пикселях
    (так как Web Mercator растягивает к полюсам), но для Астаны на zoom 19
    разница пренебрежимо мала (lat 51°). Альтернатива — делить по tile-xy
    floats — была бы математически точнее, но сложнее, а Layer 1 (capture_bbox)
    всё равно округляет к границам тайлов внутри.

    Validation бросает ValueError если bbox перевёрнут или zoom вне 1..19 —
    то же поведение что у capture_bbox, чтобы /api/scan_region и /api/capture_from_map
    давали одинаковые ошибки.
    """
    # Зум-границу берём по максимуму всех провайдеров (Esri=19, Google=20):
    # модуль не знает какой provider будет использован, провайдер-специфичный
    # check делает уже capture_bbox внутри Layer 1.
    max_z = max(p["max_zoom"] for p in TILE_PROVIDERS.values())
    if zoom < 1 or zoom > max_z:
        raise ValueError(f"zoom must be 1..{max_z}, got {zoom}")
    if nw_lat <= se_lat or nw_lng >= se_lng:
        raise ValueError(
            f"NW must be top-left of SE; got NW=({nw_lat},{nw_lng}) SE=({se_lat},{se_lng})"
        )
    if max_tiles_per_sub < 1 or max_tiles_per_sub > MAX_TILES:
        raise ValueError(
            f"max_tiles_per_sub must be in 1..{MAX_TILES} (MAX_TILES of capture_bbox), got {max_tiles_per_sub}"
        )

    total_tiles = count_tiles_for_bbox(nw_lat, nw_lng, se_lat, se_lng, zoom)
    if total_tiles <= max_tiles_per_sub:
        return [SubRegion(
            row=0, col=0,
            nw_lat=nw_lat, nw_lng=nw_lng, se_lat=se_lat, se_lng=se_lng,
        )]

    n = max(1, math.ceil(math.sqrt(total_tiles / max_tiles_per_sub)))

    lat_step = (nw_lat - se_lat) / n     # NW lat > SE lat (north positive)
    lng_step = (se_lng - nw_lng) / n     # SE lng > NW lng (east positive)

    subs: list[SubRegion] = []
    for row in range(n):
        for col in range(n):
            sub_nw_lat = nw_lat - row * lat_step
            sub_se_lat = nw_lat - (row + 1) * lat_step
            sub_nw_lng = nw_lng + col * lng_step
            sub_se_lng = nw_lng + (col + 1) * lng_step
            subs.append(SubRegion(
                row=row, col=col,
                nw_lat=sub_nw_lat, nw_lng=sub_nw_lng,
                se_lat=sub_se_lat, se_lng=sub_se_lng,
            ))
    return subs


def plan_scan(
    nw_lat: float, nw_lng: float, se_lat: float, se_lng: float,
    zoom: int,
    max_tiles_per_sub: int = DEFAULT_MAX_TILES_PER_SUB,
    max_subregions: int = DEFAULT_MAX_SUBREGIONS,
) -> list[SubRegion]:
    """Высокоуровневая обёртка: дробит bbox и ругается если получилось слишком много.

    Endpoint /api/scan_region использует именно её — `split_bbox_to_subregions`
    без лимита можно вызывать самостоятельно для тестов / dry-run прогнозов.
    """
    subs = split_bbox_to_subregions(
        nw_lat, nw_lng, se_lat, se_lng, zoom, max_tiles_per_sub=max_tiles_per_sub,
    )
    if len(subs) > max_subregions:
        raise ValueError(
            f"Region too large: would produce {len(subs)} sub-regions at zoom={zoom} "
            f"(max {max_subregions}). Уменьши прямоугольник или используй меньший zoom."
        )
    return subs
