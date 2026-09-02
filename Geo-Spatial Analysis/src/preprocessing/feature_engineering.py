"""


Feature engineering utilities implemented in pure Python.

These helpers illustrate spatial and temporal feature creation without
pandas/numpy/scipy so they are easy to read, explain, and run.
They operate on a list of dict rows with keys:
- 'date': datetime.date
- 'country': str
- 'latitude': float
- 'longitude': float
- metric fields (e.g., 'Confirmed')
"""
from __future__ import annotations

import math
from datetime import date
from typing import List, Dict, Tuple

Row = Dict[str, object]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def inverse_distance_weights(coords: List[Tuple[float, float]]) -> List[List[float]]:
    """Compute normalized inverse-distance weights for each point to all others.
    Diagonal weights are zero (no self influence). If any pair has zero distance,
    place all weight on the nearest non-zero element to avoid division by zero.
    """
    n = len(coords)
    W: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        weights = []
        for j in range(n):
            if i == j:
                w = 0.0
            else:
                d = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                w = 0.0 if d <= 0 else 1.0 / d
            weights.append(w)
        s = sum(weights)
        if s == 0:
            # All zeros (e.g., identical points); assign uniform tiny weights to others
            non_self = sum(1 for j in range(n) if j != i)
            W[i] = [0.0 if j == i else 1.0 / non_self for j in range(n)]
        else:
            W[i] = [w / s for w in weights]
    return W


def calculate_spatial_lag(rows: List[Row], value_key: str) -> List[float]:
    """Spatial lag via inverse-distance weighting of the specified value_key."""
    coords = [(float(r['latitude']), float(r['longitude'])) for r in rows]
    W = inverse_distance_weights(coords)
    values = [float(r.get(value_key, 0.0)) for r in rows]
    lags: List[float] = []
    for i in range(len(rows)):
        lags.append(sum(W[i][j] * values[j] for j in range(len(rows))))
    return lags


def calculate_hotspot_distance(rows: List[Row], value_key: str, quantile: float = 0.9) -> List[float]:
    """Distance to nearest hotspot (rows above quantile of value_key).
    If no hotspots exist, returns zeros.
    """
    values = sorted(float(r.get(value_key, 0.0)) for r in rows)
    if not values:
        return [0.0] * len(rows)
    idx = max(0, min(len(values) - 1, int(len(values) * quantile)))
    threshold = values[idx]
    hotspot_coords = [
        (float(r['latitude']), float(r['longitude']))
        for r in rows if float(r.get(value_key, 0.0)) >= threshold
    ]
    if not hotspot_coords:
        return [0.0] * len(rows)
    out: List[float] = []
    for r in rows:
        lat, lon = float(r['latitude']), float(r['longitude'])
        dmin = min(haversine_km(lat, lon, hlat, hlon) for hlat, hlon in hotspot_coords)
        out.append(dmin)
    return out


def rolling_mean_by_country(rows: List[Row], value_key: str, window: int) -> List[float]:
    """Country-level rolling mean over 'date' for the given value_key."""
    # Group rows by country and sort by date
    grouped: Dict[str, List[Tuple[date, int, int]]] = {}
    for idx, r in enumerate(rows):
        grouped.setdefault(str(r['country']), []).append((r['date'], idx, int(r.get(value_key, 0))))
    for c in grouped:
        grouped[c].sort(key=lambda t: t[0])
    # Compute rolling means, then map back to original order
    result = [0.0] * len(rows)
    for c, seq in grouped.items():
        acc: List[int] = []
        for _, idx, val in seq:
            acc.append(val)
            if len(acc) > window:
                acc.pop(0)
            result[idx] = sum(acc) / len(acc)
    return result


def engineer_features(rows: List[Row]) -> List[Row]:
    """Add temporal and spatial features for the 'Confirmed' metric."""
    if not rows:
        return rows
    # Temporal rolling means
    ma7 = rolling_mean_by_country(rows, 'Confirmed', 7)
    ma14 = rolling_mean_by_country(rows, 'Confirmed', 14)
    # Spatial features (computed per date snapshot for simplicity)
    # We compute spatial features on the full list, which approximates the screenshot dashboard use case.
    lag_conf = calculate_spatial_lag(rows, 'Confirmed')
    hotspot_dist = calculate_hotspot_distance(rows, 'Confirmed')
    # Attach features
    out: List[Row] = []
    for i, r in enumerate(rows):
        new_r = dict(r)
        new_r['confirmed_ma_7d'] = ma7[i]
        new_r['confirmed_ma_14d'] = ma14[i]
        new_r['spatial_lag_confirmed'] = lag_conf[i]
        new_r['hotspot_distance_confirmed'] = hotspot_dist[i]
        out.append(new_r)
    return out