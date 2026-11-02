import pandas as pd
import pytest

from src.ml_core.preprocessing.feature_schema import FeatureSchema, FeatureSchemaError


def test_schema_preserves_actual_feature_order_and_transforms_data():
    dataframe = pd.DataFrame({"diseases": ["a", "b"], "symptom_b": [1, 0], "symptom_a": [0, 1]})
    schema = FeatureSchema.from_training_dataframe(dataframe)
    matrix = schema.transform(dataframe)
    assert schema.feature_names == ("symptom_b", "symptom_a")
    assert matrix.toarray().tolist() == [[1, 0], [0, 1]]


def test_schema_rejects_missing_or_non_binary_feature_values():
    training = pd.DataFrame({"diseases": ["a"], "symptom": [1]})
    schema = FeatureSchema.from_training_dataframe(training)
    with pytest.raises(FeatureSchemaError):
        schema.transform(pd.DataFrame({"symptom": [2]}))
    with pytest.raises(FeatureSchemaError):
        schema.transform(pd.DataFrame({"other": [1]}))
