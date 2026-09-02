"""
Simple baseline trainer implemented in pure Python to avoid compiled dependencies.

This demonstrates the training/prediction workflow without LightGBM.
It learns a per-country moving average model over the 'Confirmed' metric.
"""
from __future__ import annotations

from datetime import date
from typing import List, Dict, Tuple

Row = Dict[str, object]


class CountryMovingAverageModel:
    def __init__(self, window: int = 7):
        self.window = window
        self.country_series: Dict[str, List[Tuple[date, int]]] = {}

    def fit(self, rows: List[Row]):
        by_country: Dict[str, List[Tuple[date, int]]] = {}
        for r in rows:
            c = str(r['country'])
            by_country.setdefault(c, []).append((r['date'], int(r.get('Confirmed', 0))))
        for c, series in by_country.items():
            series.sort(key=lambda t: t[0])
            self.country_series[c] = series
        return self

    def predict(self, rows: List[Row]) -> List[float]:
        preds: List[float] = []
        for r in rows:
            c = str(r['country'])
            d = r['date']
            series = self.country_series.get(c, [])
            # Use last `window` observations up to the date
            hist = [v for dd, v in series if dd <= d]
            hist = hist[-self.window:]
            if hist:
                preds.append(sum(hist) / len(hist))
            else:
                preds.append(0.0)
        return preds


def train_model(train_rows: List[Row], window: int = 7) -> CountryMovingAverageModel:
    """Train and return the baseline model."""
    model = CountryMovingAverageModel(window=window)
    model.fit(train_rows)
    return model