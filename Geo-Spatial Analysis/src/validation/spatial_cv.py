from __future__ import annotations

from typing import Iterator, Tuple, List, Dict, Any
from datetime import date as date_cls, datetime, timedelta

# Optional imports
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None
try:
    import numpy as np  # type: ignore
except Exception:
    np = None

Row = Dict[str, Any]


def _to_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _to_date(x: Any) -> date_cls | None:
    if isinstance(x, date_cls):
        return x
    try:
        # Support ISO strings and pandas Timestamp
        if hasattr(x, 'to_pydatetime'):
            x = x.to_pydatetime()
        return datetime.fromisoformat(str(x)).date()
    except Exception:
        return None


def _quantile_bins(values: List[float | None], n_blocks: int) -> List[int]:
    """Equal-frequency bins by rank, returning indices in [0, n_blocks-1] or -1 for invalid."""
    indexed = [(i, v) for i, v in enumerate(values) if v is not None]
    if not indexed:
        return [-1] * len(values)
    # Sort by value and assign bins by rank
    indexed.sort(key=lambda t: t[1])
    m = len(indexed)
    bins = [-1] * len(values)
    for rank, (i, _) in enumerate(indexed):
        b = min(n_blocks - 1, int(rank * n_blocks / m))
        bins[i] = b
    return bins


def _create_spatial_blocks_py(rows: List[Row], n_blocks: int = 5) -> List[int]:
    xs = [_to_float(r.get('longitude')) for r in rows]
    ys = [_to_float(r.get('latitude')) for r in rows]
    x_bins = _quantile_bins(xs, n_blocks)
    y_bins = _quantile_bins(ys, n_blocks)
    blocks = []
    for xb, yb in zip(x_bins, y_bins):
        if xb >= 0 and yb >= 0:
            blocks.append(xb * n_blocks + yb)
        else:
            blocks.append(-1)
    return blocks


def _spatial_temporal_cv_py(rows: List[Row], n_spatial_blocks: int = 5, temporal_blocks: int = 3, min_train_size: int = 30) -> Iterator[Tuple[List[int], List[int]]]:
    # Parse dates and compute min/max
    dates = [_to_date(r.get('date')) for r in rows]
    valid_idx = [i for i, d in enumerate(dates) if d is not None]
    if not valid_idx:
        raise ValueError("No valid dates after cleaning")
    dmin = min(dates[i] for i in valid_idx)  # type: ignore
    dmax = max(dates[i] for i in valid_idx)  # type: ignore
    total_days = (dmax - dmin).days
    if total_days <= 0:
        raise ValueError("Invalid date range")
    block_size = max(1, total_days // temporal_blocks)
    if block_size < min_train_size:
        raise ValueError(f"Temporal block size ({block_size} days) is smaller than minimum required training size ({min_train_size} days)")

    # Spatial blocks
    blocks = _create_spatial_blocks_py(rows, n_spatial_blocks)
    # Sort indices by date for stable behavior
    ordered = sorted(valid_idx, key=lambda i: dates[i])

    for tb in range(temporal_blocks):
        val_start = dmin + timedelta(days=tb * block_size)
        val_end = val_start + timedelta(days=block_size)
        # Determine temporal membership
        temporal_mask = {i: (dates[i] >= val_start and dates[i] < val_end) for i in ordered}
        for sb in range(n_spatial_blocks):
            train_idx: List[int] = []
            val_idx: List[int] = []
            for i in ordered:
                valid_spatial = blocks[i] >= 0
                if not valid_spatial:
                    continue
                if temporal_mask[i]:
                    if blocks[i] == sb:
                        val_idx.append(i)
                else:
                    if blocks[i] != sb:
                        train_idx.append(i)
            if len(train_idx) >= min_train_size and len(val_idx) > 0:
                yield train_idx, val_idx


# -------- Pandas/Numpy implementation --------

def _create_spatial_blocks_pd(df: 'pd.DataFrame', n_blocks: int = 5):  # type: ignore
    lon = pd.to_numeric(df['longitude'], errors='coerce')
    lat = pd.to_numeric(df['latitude'], errors='coerce')
    try:
        x_bins = pd.qcut(lon, q=n_blocks, labels=False, duplicates='drop')
        y_bins = pd.qcut(lat, q=n_blocks, labels=False, duplicates='drop')
    except ValueError as e:
        raise ValueError(f"Error creating spatial bins: {str(e)}")
    x_bins = x_bins.fillna(-1).astype('int32')
    y_bins = y_bins.fillna(-1).astype('int32')
    blocks = (np.where((x_bins >= 0) & (y_bins >= 0), x_bins * n_blocks + y_bins, -1).astype('int32'))
    return blocks


def _spatial_temporal_cv_pd(df: 'pd.DataFrame', n_spatial_blocks: int = 5, temporal_blocks: int = 3, min_train_size: int = 30) -> Iterator[Tuple['np.ndarray', 'np.ndarray']]:  # type: ignore
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    if len(df) == 0:
        raise ValueError("No valid dates after cleaning")
    try:
        df['spatial_block'] = _create_spatial_blocks_pd(df, n_spatial_blocks)
    except ValueError as e:
        raise ValueError(f"Error in spatial blocking: {str(e)}")
    df = df.sort_values('date')
    total_days = (df['date'].max() - df['date'].min()).days
    if total_days <= 0:
        raise ValueError("Invalid date range")
    block_size = max(1, total_days // temporal_blocks)
    if block_size < min_train_size:
        raise ValueError(f"Temporal block size ({block_size} days) is smaller than minimum required training size ({min_train_size} days)")
    valid_spatial_mask = df['spatial_block'] >= 0
    for temp_block in range(temporal_blocks):
        val_start = df['date'].min() + timedelta(days=temp_block * block_size)
        val_end = val_start + timedelta(days=block_size)
        temporal_mask = (df['date'] >= val_start) & (df['date'] < val_end)
        for spatial_block in range(n_spatial_blocks):
            train_mask = (~temporal_mask) & valid_spatial_mask & (df['spatial_block'] != spatial_block)
            val_mask = temporal_mask & valid_spatial_mask & (df['spatial_block'] == spatial_block)
            train_idx = df[train_mask].index.values
            val_idx = df[val_mask].index.values
            if len(train_idx) >= min_train_size and len(val_idx) > 0:
                yield train_idx, val_idx


# -------- Public API --------

def create_spatial_blocks(data: 'pd.DataFrame | List[Row]', n_blocks: int = 5):  # type: ignore
    if pd is not None and hasattr(data, 'columns'):
        return _create_spatial_blocks_pd(data, n_blocks)
    elif isinstance(data, list):
        return _create_spatial_blocks_py(data, n_blocks)
    else:
        raise TypeError("Unsupported data type: expected pandas DataFrame or list of dict rows")


def spatial_temporal_cv(data: 'pd.DataFrame | List[Row]', n_spatial_blocks: int = 5, temporal_blocks: int = 3, min_train_size: int = 30):  # type: ignore
    if pd is not None and hasattr(data, 'columns'):
        return _spatial_temporal_cv_pd(data, n_spatial_blocks, temporal_blocks, min_train_size)
    elif isinstance(data, list):
        return _spatial_temporal_cv_py(data, n_spatial_blocks, temporal_blocks, min_train_size)
    else:
        raise TypeError("Unsupported data type: expected pandas DataFrame or list of dict rows")