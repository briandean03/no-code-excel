import pandas as pd
import numpy as np
from typing import Dict, Any


class DataCleaner:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.changes = []

    def remove_empty(self, axis: str = 'both'):
        before = self.df.shape
        if axis in ['rows', 'both']:
            self.df.dropna(how='all', inplace=True)
        if axis in ['cols', 'both']:
            self.df.dropna(axis=1, how='all', inplace=True)
        after = self.df.shape
        self.changes.append(f"Removed empty {axis}: {before} → {after}")
        return self

    def fill_missing(self, method: str = 'mean'):
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if self.df[col].isnull().any():
                if method == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif method == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif method == 'zero':
                    self.df[col].fillna(0, inplace=True)
        self.changes.append(f"Filled missing numeric cells using {method}")
        return self

    def standardize_dates(self):
        for col in self.df.columns:
            try:
                parsed = pd.to_datetime(self.df[col], errors='coerce')
                if parsed.notna().sum() > len(self.df) * 0.7:
                    self.df[col] = parsed.dt.strftime("%Y-%m-%d")
                    self.changes.append(f"Standardized date format in column '{col}'")
            except:
                continue
        return self

    def get_summary(self) -> Dict[str, Any]:
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "changes": self.changes
        }

    def get_cleaned_df(self):
        return self.df
