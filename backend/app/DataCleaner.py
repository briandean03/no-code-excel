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
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif method == 'median':
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                elif method == 'zero':
                    self.df[col] = self.df[col].fillna(0)
        self.changes.append(f"Filled missing numeric cells using {method}")
        return self

    def standardize_dates(self):
        """
        FIXED VERSION:
        - Only attempts parsing on object/string columns
        - Column must have 90%+ valid date parses to be considered a date
        - Prevents accidental conversion of normal text to dates
        """
        for col in self.df.columns:

            # Only parse string/object columns
            if self.df[col].dtype != object:
                continue

            # Try parsing using pandas new mixed parser
            parsed = pd.to_datetime(self.df[col], errors='coerce', format='mixed')

            # Require high confidence (> 90% valid) to treat column as date
            if parsed.notna().sum() >= len(self.df) * 0.9:
                # Convert to yyyy-mm-dd
                self.df[col] = parsed.dt.strftime("%Y-%m-%d")
                self.changes.append(f"Standardized date format in '{col}'")

        return self

    def get_summary(self) -> Dict[str, Any]:
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "changes": self.changes
        }

    def get_cleaned_df(self):
        return self.df
