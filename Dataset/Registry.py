import pandas as pd
from uuid import uuid4

class DatasetRegistry:
    def __init__(self):
        self._store: dict[str, pd.DataFrame] = {}

    def register(self, df) -> str:
        dataset_id = str(uuid4())
        self._store[dataset_id] = df
        return dataset_id

    def get(self, dataset_id: str) -> pd.DataFrame:
        if dataset_id not in self._store:
            raise KeyError(f"Dataset {dataset_id} not found")
        return self._store[dataset_id]

    def delete(self, dataset_id: str):
        self._store.pop(dataset_id, None)

dataset_registry = DatasetRegistry()
