"""Phase 3: xác thực và chuyển đổi feature từ DataFrame dữ liệu thực."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from scipy import sparse


class FeatureSchemaError(ValueError):
    """Dữ liệu không tuân theo feature schema đã xác minh."""


@dataclass(frozen=True)
class FeatureSchema:
    target_column: str
    feature_names: tuple[str, ...]
    version: str = "1.0"

    @classmethod
    def from_training_dataframe(
        cls, dataframe: pd.DataFrame, target_column: str = "diseases"
    ) -> "FeatureSchema":
        if target_column not in dataframe.columns:
            raise FeatureSchemaError(f"Không có cột target '{target_column}'.")
        features = tuple(column for column in dataframe.columns if column != target_column)
        if not features:
            raise FeatureSchemaError("Dataset không có feature.")
        schema = cls(target_column=target_column, feature_names=features)
        schema.validate_dataframe(dataframe, require_target=True)
        return schema

    def validate_dataframe(self, dataframe: pd.DataFrame, require_target: bool) -> None:
        if require_target and self.target_column not in dataframe.columns:
            raise FeatureSchemaError(f"Thiếu target '{self.target_column}'.")
        actual_features = [column for column in dataframe.columns if column != self.target_column]
        if actual_features != list(self.feature_names):
            raise FeatureSchemaError("Feature names hoặc thứ tự feature không khớp schema.")
        features = dataframe.loc[:, self.feature_names]
        if features.isna().any().any():
            raise FeatureSchemaError("Feature có giá trị thiếu.")
        invalid_columns = [
            column for column in self.feature_names
            if not features[column].isin([0, 1]).all()
        ]
        if invalid_columns:
            raise FeatureSchemaError(
                f"Feature phải nhị phân 0/1. Cột không hợp lệ: {invalid_columns[:5]}"
            )

    def transform(self, dataframe: pd.DataFrame) -> sparse.csr_matrix:
        """Chuyển feature của DataFrame thực thành CSR, không dùng metadata làm dữ liệu."""
        self.validate_dataframe(dataframe, require_target=False)
        return sparse.csr_matrix(dataframe.loc[:, self.feature_names].to_numpy(dtype="int8"))

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "target_column": self.target_column,
            "feature_count": len(self.feature_names),
            "feature_names": list(self.feature_names),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
