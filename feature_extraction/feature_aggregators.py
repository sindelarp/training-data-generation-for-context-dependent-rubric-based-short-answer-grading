from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class FeatureAggregator(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def add(self, feature_dict: dict):
        pass

    @abstractmethod
    def aggregate(self) -> Any:
        pass


class MeanAggregator(FeatureAggregator):
    def __init__(self, feature_name: str):
        self._feature_name = feature_name
        self._count = 0
        self._sum = 0

    @property
    def name(self) -> str:
        return f"{self._feature_name}_mean"

    def add(self, feature_dict: dict):
        self._sum += feature_dict[self._feature_name]
        self._count += 1

    def aggregate(self) -> Any:
        if self._count == 0:
            return 0
        return self._sum / self._count


class PercentileAggregator(FeatureAggregator):
    def __init__(self, feature_name: str, percentiles: list[float]):
        self._feature_name = feature_name
        self._percentiles = sorted(percentiles)
        self._values = []

    @property
    def name(self) -> str:
        return f"{self._feature_name}_percentiles"

    def add(self, feature_dict: dict):
        self._values.append(feature_dict[self._feature_name])

    def aggregate(self) -> Any:
        if not self._values:
            return {str(p): 0 for p in self._percentiles}
        return np.quantile(self._values, [p / 100 for p in self._percentiles]).tolist()


class PercentileMatrixAggregator(FeatureAggregator):
    def __init__(
            self, feature_name_1: str, feature_name_2: str, percentiles: list[float]
    ):
        self._feature_name_1 = feature_name_1
        self._feature_name_2 = feature_name_2
        self._percentiles = sorted(percentiles)
        self._values = []

    @property
    def name(self) -> str:
        return f"{self._feature_name_1}_{self._feature_name_2}_percentile_matrix"

    def add(self, feature_dict: dict):
        self._values.append(
            (feature_dict[self._feature_name_1], feature_dict[self._feature_name_2])
        )

    def _get_single_percentile(
            self, sorted_data: list[float], percentile: float
    ) -> float:
        if not sorted_data:
            return 0

        n = len(sorted_data)
        float_position = (n - 1) * (percentile / 100)
        position = int(float_position)
        bounded_position = min(position + 1, n - 1)

        if position == bounded_position:
            return sorted_data[position]
        else:
            d0 = sorted_data[position] * (bounded_position - float_position)
            d1 = sorted_data[bounded_position] * (float_position - position)
            return d0 + d1

    def aggregate(self) -> Any:
        if not self._values:
            return {
                {
                    str(p): {str(p2): 0 for p2 in self._percentiles}
                    for p in self._percentiles
                }
            }
        x_values = sorted([v[0] for v in self._values])
        matrix = {}
        for p_x in self._percentiles:
            x_threshold = self._get_single_percentile(x_values, p_x)
            subset_y = [v[1] for v in self._values if v[0] <= x_threshold]
            subset_y_sorted = sorted(subset_y)
            row_results = {}
            for p_y in self._percentiles:
                row_results[str(p_y)] = self._get_single_percentile(
                    subset_y_sorted, p_y
                )
            matrix[str(p_x)] = row_results
        return matrix
