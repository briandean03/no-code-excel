import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from scipy.ndimage import label
import logging

class TableProcessor:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    # ========= PUBLIC API =========
    def detect_tables(
        self,
        df: pd.DataFrame,
        min_data_density: float = 0.3,
        min_cols: int = 2,
        min_rows: int = 3,
        max_gap: int = 3,
    ) -> List[Dict]:
        """
        Returns list of dicts: {table: DataFrame, start_row, end_row, ...}
        """
        if df.empty:
            self.logger.info("DataFrame is empty")
            return []

        original_shape = df.shape
        df_clean = df.dropna(how='all').dropna(axis=1, how='all')
        self.logger.info(f"Initial cleanup: {original_shape} -> {df_clean.shape}")

        if df_clean.empty:
            self.logger.info("DataFrame empty after cleanup")
            return []

        tables = []
        def _is_filled_cell(v):
            if pd.isna(v):
                return False
            if isinstance(v, str) and v.strip() == "":
                return False
            return True

        data_mask = df_clean.applymap(_is_filled_cell).astype(int)
        structure = np.array([[1,1,1],[1,1,1],[1,1,1]])
        labeled_array, num_features = label(data_mask.values, structure=structure)
        self.logger.info(f"Found {num_features} potential data regions")

        for label_id in range(1, num_features + 1):
            positions = np.argwhere(labeled_array == label_id)
            if positions.size == 0:
                continue
            min_row, min_col = positions.min(axis=0)
            max_row, max_col = positions.max(axis=0) + 1
            block_df = df_clean.iloc[min_row:max_row, min_col:max_col].copy()

            if block_df.shape[0] < min_rows or block_df.shape[1] < min_cols:
                continue

            sub_tables = self._split_sub_tables(
                block_df, min_data_density, min_rows, min_cols, max_gap
            )
            for sub_table, sub_metadata in sub_tables:
                tables.append({
                    "table": sub_table,
                    "sheet_name": "Sheet 1",
                    "header_row": min_row + sub_metadata.get("header_idx")
                                  if sub_metadata.get("header_idx") is not None else None,
                    "start_row": min_row + sub_metadata["start_row"],
                    "end_row": min_row + sub_metadata["end_row"],
                    "start_col": min_col + sub_metadata["start_col"],
                    "end_col": min_col + sub_metadata["end_col"],
                    "confidence_score": sub_metadata["confidence"],
                    "is_tabular": sub_metadata["is_tabular"]
                })

        tables = self._filter_and_deduplicate_tables_dicts(tables, min_cols, min_rows)
        self.logger.info(f"Detected {len(tables)} robust tables")
        return tables

    # ========= INTERNALS (your original functions, lightly adapted) =========

    def _split_sub_tables(self, df: pd.DataFrame, min_data_density: float, min_rows: int,
                          min_cols: int, max_gap: int) -> List[Tuple[pd.DataFrame, Dict]]:
        sub_tables = []
        sections = self._find_data_sections_flexible(df, min_data_density, min_rows, max_gap)

        for start_idx, end_idx in sections:
            section_df = df.iloc[start_idx:end_idx].copy()
            is_tabular, confidence = self._is_tabular_section(section_df, min_data_density)

            if not is_tabular:
                cleaned = self._clean_non_tabular_data(section_df, min_data_density)
                if cleaned is not None and len(cleaned) >= min_rows and len(cleaned.columns) >= min_cols:
                    sub_tables.append((cleaned, {
                        "header_idx": None, "start_row": start_idx, "end_row": end_idx,
                        "start_col": 0, "end_col": len(df.columns),
                        "confidence": confidence * 0.5, "is_tabular": False
                    }))
                continue

            header_rows = self._find_header_rows_in_section(section_df)
            if header_rows:
                for header_idx in header_rows[:2]:
                    tbl, conf = self._extract_table_with_metadata(
                        section_df, header_idx, min_data_density, min_cols, min_rows
                    )
                    if tbl is not None and len(tbl) >= min_rows:
                        sub_tables.append((tbl, {
                            "header_idx": header_idx,
                            "start_row": header_idx + 1,
                            "end_row": header_idx + 1 + len(tbl),
                            "start_col": 0,
                            "end_col": len(section_df.columns),
                            "confidence": conf,
                            "is_tabular": True
                        }))
            else:
                cleaned = self._clean_table_advanced(section_df, min_data_density)
                if cleaned is not None and len(cleaned) >= min_rows and len(cleaned.columns) >= min_cols:
                    sub_tables.append((cleaned, {
                        "header_idx": None, "start_row": start_idx, "end_row": end_idx,
                        "start_col": 0, "end_col": len(section_df.columns),
                        "confidence": 0.5, "is_tabular": True
                    }))
        return sub_tables

    def _is_tabular_section(self, df: pd.DataFrame, min_data_density: float) -> Tuple[bool, float]:
        if df.empty:
            return False, 0.0
        density = df.notna().sum().sum() / (len(df) * len(df.columns))
        if density < min_data_density:
            return False, density
        col_consistency = 0
        for col in df.columns:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                type_ratio = len(set(non_null.apply(lambda x: type(x).__name__))) / len(non_null)
                col_consistency += max(0, 1 - type_ratio)
        col_consistency = col_consistency / len(df.columns) if len(df.columns) > 0 else 0

        row_similarity = 0
        for i in range(1, min(5, len(df))):
            row1, row2 = df.iloc[i-1], df.iloc[i]
            matches = sum(1 for v1, v2 in zip(row1, row2) if pd.isna(v1) == pd.isna(v2) or str(v1) == str(v2))
            row_similarity += matches / len(row1) if len(row1) > 0 else 0
        row_similarity = row_similarity / (min(4, len(df)-1)) if len(df) > 1 else 0

        confidence = (density * 0.4 + col_consistency * 0.3 + row_similarity * 0.3)
        return confidence > 0.5, confidence

    def _clean_non_tabular_data(self, df: pd.DataFrame, min_data_density: float) -> Optional[pd.DataFrame]:
        if df.empty:
            return None
        flat_data = []
        for _, row in df.iterrows():
            non_null = row.dropna().astype(str).str.strip()
            if len(non_null) >= 2:
                flat_data.append(non_null.values.tolist())
        if not flat_data:
            return None
        max_cols = max(len(row) for row in flat_data)
        clean_df = pd.DataFrame(flat_data, columns=[f"Field_{i+1}" for i in range(max_cols)])
        clean_df = clean_df.dropna(how='all').dropna(axis=1, how='all')
        if clean_df.empty or len(clean_df) < 2 or len(clean_df.columns) < 2:
            return None
        return clean_df

    def _find_data_sections_flexible(self, df: pd.DataFrame, min_density: float, min_rows: int,
                                     max_gap: int = 3) -> List[Tuple[int, int]]:
        row_densities = [row.notna().sum() / len(row) if len(row) > 0 else 0 for _, row in df.iterrows()]
        sections, start_idx, gap_count = [], None, 0
        for i, density in enumerate(row_densities):
            if density >= min_density:
                if start_idx is None:
                    start_idx = i
                gap_count = 0
            else:
                if start_idx is not None:
                    gap_count += 1
                    if gap_count > max_gap:
                        if i - gap_count - start_idx >= min_rows:
                            sections.append((start_idx, i - gap_count))
                        start_idx, gap_count = None, 0
        if start_idx is not None and len(row_densities) - start_idx >= min_rows:
            sections.append((start_idx, len(row_densities)))
        return sections

    def _find_header_rows_in_section(self, df: pd.DataFrame) -> List[int]:
        header_candidates = []
        search_range = min(10, len(df))
        for i in range(search_range):
            row = df.iloc[i]
            non_null_values = row.dropna()
            if len(non_null_values) < 2:
                continue
            text_values = [str(val).strip() for val in non_null_values if str(val).strip()]
            if len(text_values) >= 2:
                score = self._calculate_header_score_advanced(text_values, i, len(df))
                if score > 0.5:
                    header_candidates.append((i, score))
        header_candidates.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in header_candidates[:3]]

    def _calculate_header_score_advanced(self, values: List[str], row_idx: int, total_rows: int) -> float:
        if not values:
            return 0.0
        score, total = 0.0, len(values)
        cleaned = [v for v in values if v and v.strip()]
        if not cleaned:
            return 0.0
        text_count = sum(1 for v in cleaned if not self._is_numeric_string(v))
        score += (text_count / total) * 0.3
        good_length_count = sum(1 for v in cleaned if 2 <= len(v) <= 50)
        score += (good_length_count / total) * 0.2
        score += (len(set(cleaned)) / total) * 0.25
        score += max(0, (10 - row_idx) / 10) * 0.15
        alpha_count = sum(1 for v in cleaned if re.search(r'[a-zA-Z]', v))
        score += (alpha_count / total) * 0.1
        return max(0, min(score, 1.0))

    def _is_numeric_string(self, s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _extract_table_with_metadata(self, df: pd.DataFrame, header_idx: int, min_data_density: float,
                                     min_cols: int, min_rows: int) -> Tuple[Optional[pd.DataFrame], float]:
        if header_idx >= len(df):
            return None, 0.0
        header_row = df.iloc[header_idx]
        data_start = header_idx + 1
        table_end = self._find_table_end_smart(df, data_start, min_data_density)
        if table_end - data_start < min_rows:
            return None, 0.0
        raw_table = df.iloc[data_start:table_end].copy()
        clean_headers = self._process_headers(header_row, len(raw_table.columns))
        raw_table.columns = clean_headers
        # treat whitespace as empty
        raw_table = raw_table.applymap(lambda x: (np.nan if (isinstance(x, str) and x.strip() == "") else x))

        # drop columns that are entirely empty
        raw_table = raw_table.dropna(axis=1, how='all')

        # strip empty EDGE columns based on a small non-null threshold
        non_null_counts = raw_table.notna().sum()
        min_non_null_rows = max(2, int(0.2 * len(raw_table)))  # at least 2 rows with data
        valid_cols = [i for i, c in enumerate(non_null_counts) if c >= min_non_null_rows]
        if valid_cols:
            left, right = min(valid_cols), max(valid_cols)
            raw_table = raw_table.iloc[:, left:right+1]
        # Drop trailing all-empty columns BEFORE cleaning
        raw_table = raw_table.dropna(axis=1, how='all')
        cleaned = self._clean_table_advanced(raw_table, min_data_density)

        if cleaned is None or len(cleaned) < min_rows or len(cleaned.columns) < min_cols:
            return None, 0.0
        header_values = [str(x).strip() for x in header_row if str(x).strip()]
        header_score = self._calculate_header_score_advanced(header_values, header_idx, len(df))
        data_density = cleaned.notna().sum().sum() / (len(cleaned) * len(cleaned.columns))
        confidence = 0.6 * header_score + 0.4 * data_density
        return cleaned, confidence

    def _find_table_end_smart(self, df: pd.DataFrame, start_idx: int, min_data_density: float) -> int:
        consecutive_empty, last_good_row = 0, start_idx
        for i in range(start_idx, len(df)):
            row = df.iloc[i]
            non_null_ratio = row.notna().sum() / len(row) if len(row) > 0 else 0
            if non_null_ratio < min_data_density:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    return last_good_row + 1
            else:
                consecutive_empty = 0
                last_good_row = i
        return len(df)

    def _process_headers(self, header_row: pd.Series, num_cols: int) -> List[str]:
        clean_headers, seen = [], {}
        for i in range(num_cols):
            if i < len(header_row):
                header_val = header_row.iloc[i]
                if pd.isna(header_val) or str(header_val).strip() == '':
                    header = f"Column_{i+1}"
                else:
                    header = str(header_val).strip()
                    header = re.sub(r'[^\w\s-]', '', header)
                    header = re.sub(r'\s+', '_', header) or f"Column_{i+1}"
            else:
                header = f"Column_{i+1}"
            if header in seen:
                seen[header] += 1
                header = f"{header}_{seen[header]}"
            else:
                seen[header] = 1
            clean_headers.append(header)
        return clean_headers

    def _clean_table_advanced(self, df: pd.DataFrame, min_data_density: float = 0.3) -> Optional[pd.DataFrame]:
        if df.empty:
            return None
        df = df.apply(lambda x: x.astype(str).str.lower().replace('nan', np.nan) if x.dtype == 'object' else x)
        df = df.replace(['', 'None'], np.nan)
        df = df.dropna(how='all').dropna(axis=1, how='all')
                                        
        df = df.applymap(lambda x: (np.nan if (isinstance(x, str) and x.strip() == "") else x))
        df = df.dropna(axis=1, how='all')

        # Drop ultra-sparse columns (< 10% non-null)
        col_non_null_ratio = df.notna().sum() / max(1, len(df))
        df = df.loc[:, col_non_null_ratio >= max(0.1, min_data_density * 0.5)]
        # Drop columns that are all empty strings
        df = df.loc[:, (df != '').any(axis=0)]

        if df.empty:
            return None
        df = df.dropna(thresh=max(1, int(len(df.columns) * min_data_density)))
        if df.empty:
            return None
        df = df.reset_index(drop=True)
        for col in df.columns:
            df[col] = self._infer_and_convert_column_type(df[col])
        return df

    def _infer_and_convert_column_type(self, series: pd.Series) -> pd.Series:
        if series.dtype == 'object':
            series = series.astype(str).str.strip()
            series = series.replace(['nan', 'NaN', 'none', 'NONE', ''], np.nan)

        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return series

        numeric_count = sum(1 for val in non_null_series if self._is_numeric_string(str(val)))
        numeric_ratio = numeric_count / len(non_null_series)
        if numeric_ratio > 0.85:
            try:
                numeric_series = pd.to_numeric(series, errors='coerce')
                if numeric_series.notna().sum() >= len(non_null_series) * 0.8:
                    return numeric_series
            except:
                pass

        date_pattern = r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}'
        date_count = sum(1 for val in non_null_series.head(10) if re.search(date_pattern, str(val)))
        if date_count / min(10, len(non_null_series)) > 0.7:
            try:
                date_series = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
                if date_series.notna().sum() >= len(non_null_series) * 0.7:
                    return date_series
            except:
                pass
        return series

    def _filter_and_deduplicate_tables_dicts(self, tables: List[Dict], min_cols: int, min_rows: int) -> List[Dict]:
        filtered = [t for t in tables if len(t["table"]) >= min_rows and len(t["table"].columns) >= min_cols]
        unique_tables: List[Tuple[Dict, Tuple[int,int,int,int]]] = []
        for t in filtered:
            is_dup = False
            current_range = (t["start_row"], t["end_row"], t["start_col"], t["end_col"])
            for ut, ut_range in unique_tables:
                if (max(current_range[0], ut_range[0]) < min(current_range[1], ut_range[1]) and
                    max(current_range[2], ut_range[2]) < min(current_range[3], ut_range[3])):
                    if self._tables_are_similar(t["table"], ut["table"], similarity_threshold=0.9):
                        is_dup = True
                        break
            if not is_dup:
                unique_tables.append((t, current_range))
        return [t for t, _ in unique_tables]

    def _tables_are_similar(self, table1: pd.DataFrame, table2: pd.DataFrame, similarity_threshold: float = 0.9) -> bool:
        if table1.shape != table2.shape:
            return False
        sample_size = min(5, len(table1))
        try:
            content_matches = 0
            total_cells = sample_size * len(table1.columns)
            for i in range(sample_size):
                for j in range(len(table1.columns)):
                    val1, val2 = table1.iloc[i, j], table2.iloc[i, j]
                    if (pd.isna(val1) and pd.isna(val2)) or (str(val1).strip() == str(val2).strip()):
                        content_matches += 1
            return (content_matches / total_cells) >= similarity_threshold
        except:
            return False

    def get_table_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {}
        df_clean = df.apply(lambda x: x.astype(str).str.lower().replace('nan', np.nan) if x.dtype == 'object' else x)
        df_clean = df_clean.replace(['', 'None'], np.nan)
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_data': df.isnull().sum().to_dict(),
            'non_null_counts': df.notna().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'sample_data': df_clean.head(10).replace({np.nan: None}).to_dict(orient='records'),
            'data_quality': {
                'completeness': (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'unique_values_per_column': {col: df[col].nunique() for col in df.columns}
            }
        }
        return info
