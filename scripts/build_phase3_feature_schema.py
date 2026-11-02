"""Tạo artifact Phase 3 từ CSV train thực tế của Phase 2; không đọc metadata thí nghiệm."""

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_core.preprocessing.feature_schema import FeatureSchema


def main() -> None:
    train_path = PROJECT_ROOT / "data" / "processed" / "exp1_full" / "train.csv"
    output_path = PROJECT_ROOT / "docs" / "phase3_feature_schema.json"
    train_dataframe = pd.read_csv(train_path)
    schema = FeatureSchema.from_training_dataframe(train_dataframe)
    schema.save(output_path)
    print(f"Saved {len(schema.feature_names)}-feature schema to {output_path}")


if __name__ == "__main__":
    main()
