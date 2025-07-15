import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Any
from scipy.ndimage import label
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_and_clean_tables(df: pd.DataFrame, min_data_density: float = 0.3,
                          min_cols: int = 2, min_rows: int = 3, max_gap: int = 3) -> List[Dict]:
    """
    Detect and clean multiple tables in messy Excel sheets with no hardcoded assumptions.
    Returns a list of dicts: each dict contains 'table': DataFrame and metadata.
    """
    if df.empty:
        logger.info("DataFrame is empty")
        return []

    original_shape = df.shape
    df_clean = df.dropna(how='all').dropna(axis=1, how='all')
    logger.info(f"Initial cleanup: {original_shape} -> {df_clean.shape}")

    if df_clean.empty:
        logger.info("DataFrame empty after cleanup")
        return []

    tables = []

    # Step 1: Convert to binary grid: 1 = data, 0 = empty
    data_mask = df_clean.notna().astype(int)

    # Step 2: Identify data clusters using connected component labeling
    structure = np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]])
    labeled_array, num_features = label(data_mask.values, structure=structure)
    logger.info(f"Found {num_features} potential data regions")

    # Step 3: Process each data cluster
    for label_id in range(1, num_features + 1):
        positions = np.argwhere(labeled_array == label_id)
        if positions.size == 0:
            continue

        min_row, min_col = positions.min(axis=0)
        max_row, max_col = positions.max(axis=0) + 1
        block_df = df_clean.iloc[min_row:max_row, min_col:max_col].copy()

        if block_df.shape[0] < min_rows or block_df.shape[1] < min_cols:
            continue

        # Step 4: Split potential stacked or side-by-side tables within the block
        sub_tables = split_sub_tables(block_df, min_data_density, min_rows, min_cols, max_gap)
        
        for sub_table, sub_metadata in sub_tables:
            tables.append({
                "table": sub_table,
                "sheet_name": "Sheet 1",
                "header_row": min_row + sub_metadata["header_idx"],
                "start_row": min_row + sub_metadata["start_row"],
                "end_row": min_row + sub_metadata["end_row"],
                "start_col": min_col + sub_metadata["start_col"],
                "end_col": min_col + sub_metadata["end_col"],
                "confidence_score": sub_metadata["confidence"],
                "is_tabular": sub_metadata["is_tabular"]
            })

    # Step 5: Filter out non-tabular regions and deduplicate
    tables = filter_and_deduplicate_tables_dicts(tables, min_cols, min_rows)
    logger.info(f"Detected {len(tables)} robust tables")
    return tables

def split_sub_tables(df: pd.DataFrame, min_data_density: float, min_rows: int, 
                    min_cols: int, max_gap: int) -> List[Tuple[pd.DataFrame, Dict]]:
    """
    Split a DataFrame block into sub-tables based on gaps and density.
    Returns list of (sub_table, metadata) tuples.
    """
    sub_tables = []
    row_densities = [row.notna().sum() / len(row) if len(row) > 0 else 0 for _, row in df.iterrows()]

    # Find continuous data sections
    sections = find_data_sections_flexible(df, min_data_density, min_rows, max_gap)

    for start_idx, end_idx in sections:
        section_df = df.iloc[start_idx:end_idx].copy()
        
        # Check if section is tabular
        is_tabular, confidence = is_tabular_section(section_df, min_data_density)
        if not is_tabular:
            # Try to clean non-tabular data (e.g., form responses, logs)
            cleaned = clean_non_tabular_data(section_df, min_data_density)
            if cleaned is not None and len(cleaned) >= min_rows and len(cleaned.columns) >= min_cols:
                sub_tables.append((cleaned, {
                    "header_idx": None,
                    "start_row": start_idx,
                    "end_row": end_idx,
                    "start_col": 0,
                    "end_col": len(df.columns),
                    "confidence": confidence * 0.5,
                    "is_tabular": False
                }))
            continue

        # Detect headers in tabular section
        header_rows = find_header_rows_in_section(section_df)
        
        if header_rows:
            for header_idx in header_rows[:2]:  # Limit to top 2 headers to avoid over-splitting
                tbl, conf = extract_table_with_metadata(section_df, header_idx, min_data_density, min_cols, min_rows)
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
            # No clear header, treat as headerless table
            cleaned = clean_table_advanced(section_df, min_data_density)
            if cleaned is not None and len(cleaned) >= min_rows and len(cleaned.columns) >= min_cols:
                sub_tables.append((cleaned, {
                    "header_idx": None,
                    "start_row": start_idx,
                    "end_row": end_idx,
                    "start_col": 0,
                    "end_col": len(section_df.columns),
                    "confidence": 0.5,
                    "is_tabular": True
                }))

    return sub_tables

def is_tabular_section(df: pd.DataFrame, min_data_density: float) -> Tuple[bool, float]:
    """
    Determine if a DataFrame section is tabular based on structure and content.
    Returns (is_tabular, confidence_score).
    """
    if df.empty:
        return False, 0.0

    # Calculate data density
    density = df.notna().sum().sum() / (len(df) * len(df.columns))
    if density < min_data_density:
        return False, density

    # Check column consistency (similar data types or patterns across rows)
    col_consistency = 0
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) > 0:
            type_ratio = len(set(non_null.apply(lambda x: type(x).__name__)))/len(non_null)
            col_consistency += max(0, 1 - type_ratio)
    col_consistency = col_consistency / len(df.columns) if len(df.columns) > 0 else 0

    # Check for repeating patterns (indicative of tabular data)
    row_similarity = 0
    for i in range(1, min(5, len(df))):
        row1, row2 = df.iloc[i-1], df.iloc[i]
        matches = sum(1 for v1, v2 in zip(row1, row2) if pd.isna(v1) == pd.isna(v2) or str(v1) == str(v2))
        row_similarity += matches / len(row1) if len(row1) > 0 else 0
    row_similarity = row_similarity / (min(4, len(df)-1)) if len(df) > 1 else 0

    confidence = (density * 0.4 + col_consistency * 0.3 + row_similarity * 0.3)
    return confidence > 0.5, confidence

def clean_non_tabular_data(df: pd.DataFrame, min_data_density: float) -> Optional[pd.DataFrame]:
    """
    Attempt to structure non-tabular data (e.g., form responses, logs) into a table.
    """
    if df.empty:
        return None

    # Flatten non-tabular data into key-value pairs if possible
    flat_data = []
    for i, row in df.iterrows():
        non_null = row.dropna().astype(str).str.strip()
        if len(non_null) >= 2:
            flat_data.append(non_null.values.tolist())

    if not flat_data:
        return None

    # Create a DataFrame from flattened data
    max_cols = max(len(row) for row in flat_data)
    clean_df = pd.DataFrame(flat_data, columns=[f"Field_{i+1}" for i in range(max_cols)])
    clean_df = clean_df.dropna(how='all').dropna(axis=1, how='all')
    
    if clean_df.empty or len(clean_df) < 2 or len(clean_df.columns) < 2:
        return None

    return clean_df

def find_data_sections_flexible(df: pd.DataFrame, min_density: float, min_rows: int,
                              max_gap: int = 3) -> List[Tuple[int, int]]:
    """
    Find continuous data sections, merging sections separated by small gaps.
    """
    row_densities = [row.notna().sum() / len(row) if len(row) > 0 else 0 for _, row in df.iterrows()]
    sections = []
    start_idx = None
    gap_count = 0

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
                    start_idx = None
                    gap_count = 0

    if start_idx is not None and len(row_densities) - start_idx >= min_rows:
        sections.append((start_idx, len(row_densities)))

    return sections

def find_header_rows_in_section(df: pd.DataFrame) -> List[int]:
    """
    Detect potential header rows based on content patterns, no hardcoded names.
    """
    header_candidates = []
    search_range = min(10, len(df))

    for i in range(search_range):
        row = df.iloc[i]
        non_null_values = row.dropna()

        if len(non_null_values) < 2:
            continue

        text_values = [str(val).strip() for val in non_null_values if str(val).strip()]
        if len(text_values) >= 2:
            score = calculate_header_score_advanced(text_values, i, len(df))
            if score > 0.5:
                header_candidates.append((i, score))

    header_candidates.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, score in header_candidates[:3]]

def calculate_header_score_advanced(values: List[str], row_idx: int, total_rows: int) -> float:
    """
    Score potential header row based on content characteristics.
    """
    if not values:
        return 0.0

    score = 0.0
    total = len(values)
    cleaned = [v for v in values if v and v.strip()]

    if not cleaned:
        return 0.0

    # Factor 1: Text vs numeric ratio (headers are typically text)
    text_count = sum(1 for v in cleaned if not is_numeric_string(v))
    text_ratio = text_count / total
    score += text_ratio * 0.3

    # Factor 2: Length distribution (headers have moderate length)
    good_length_count = sum(1 for v in cleaned if 2 <= len(v) <= 50)
    length_ratio = good_length_count / total
    score += length_ratio * 0.2

    # Factor 3: Uniqueness (headers should be distinct)
    unique_ratio = len(set(cleaned)) / total
    score += unique_ratio * 0.25

    # Factor 4: Position (headers are often near the top)
    position_bonus = max(0, (10 - row_idx) / 10) * 0.15
    score += position_bonus

    # Factor 5: Alphanumeric content (headers often have letters)
    alpha_count = sum(1 for v in cleaned if re.search(r'[a-zA-Z]', v))
    alpha_ratio = alpha_count / total
    score += alpha_ratio * 0.1

    return max(0, min(score, 1.0))

def is_numeric_string(s: str) -> bool:
    """
    Check if string represents a number.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

def extract_table_with_metadata(df: pd.DataFrame, header_idx: int, min_data_density: float,
                              min_cols: int, min_rows: int) -> Tuple[Optional[pd.DataFrame], float]:
    """
    Extract table starting from header row, with confidence score.
    """
    if header_idx >= len(df):
        return None, 0.0

    header_row = df.iloc[header_idx]
    data_start = header_idx + 1
    table_end = find_table_end_smart(df, data_start, min_data_density)

    if table_end - data_start < min_rows:
        return None, 0.0

    raw_table = df.iloc[data_start:table_end].copy()
    clean_headers = process_headers(header_row, len(raw_table.columns))
    raw_table.columns = clean_headers

    cleaned = clean_table_advanced(raw_table, min_data_density)
    if cleaned is None or len(cleaned) < min_rows or len(cleaned.columns) < min_cols:
        return None, 0.0

    header_values = [str(x).strip() for x in header_row if str(x).strip()]
    header_score = calculate_header_score_advanced(header_values, header_idx, len(df))
    data_density = cleaned.notna().sum().sum() / (len(cleaned) * len(cleaned.columns))
    confidence = 0.6 * header_score + 0.4 * data_density

    return cleaned, confidence

def find_table_end_smart(df: pd.DataFrame, start_idx: int, min_data_density: float) -> int:
    """
    Detect table end based on density and gaps.
    """
    consecutive_empty = 0
    last_good_row = start_idx

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

def process_headers(header_row: pd.Series, num_cols: int) -> List[str]:
    """
    Generate clean, unique column names from header row.
    """
    clean_headers = []
    seen = {}

    for i in range(num_cols):
        if i < len(header_row):
            header_val = header_row.iloc[i]
            if pd.isna(header_val) or str(header_val).strip() == '':
                header = f"Column_{i+1}"
            else:
                header = str(header_val).strip()
                header = re.sub(r'[^\w\s-]', '', header)
                header = re.sub(r'\s+', '_', header)
                header = header if header else f"Column_{i+1}"
        else:
            header = f"Column_{i+1}"

        if header in seen:
            seen[header] += 1
            header = f"{header}_{seen[header]}"
        else:
            seen[header] = 1
        clean_headers.append(header)

    return clean_headers

def clean_table_advanced(df: pd.DataFrame, min_data_density: float = 0.3) -> Optional[pd.DataFrame]:
    """
    Clean table by removing low-density rows/columns and inferring types.
    """
    if df.empty:
        return None

    # Replace case-insensitive 'nan' strings with np.nan
    df = df.apply(lambda x: x.astype(str).str.lower().replace('nan', np.nan) if x.dtype == 'object' else x)
    df = df.replace(['', 'None'], np.nan)
    df = df.dropna(how='all').dropna(axis=1, how='all')
    if df.empty:
        return None

    df = df.dropna(thresh=max(1, int(len(df.columns) * min_data_density)))
    if df.empty:
        return None

    df = df.reset_index(drop=True)
    for col in df.columns:
        df[col] = infer_and_convert_column_type(df[col])

    return df

def infer_and_convert_column_type(series: pd.Series) -> pd.Series:
    """
    Infer and convert column data types dynamically.
    """
    if series.dtype == 'object':
        series = series.astype(str).str.strip()
        series = series.replace(['nan', 'NaN', 'none', 'NONE', ''], np.nan)

    non_null_series = series.dropna()
    if len(non_null_series) == 0:
        return series

    # Numeric detection
    numeric_count = sum(1 for val in non_null_series if is_numeric_string(str(val)))
    numeric_ratio = numeric_count / len(non_null_series)

    if numeric_ratio > 0.85:
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            if numeric_series.notna().sum() >= len(non_null_series) * 0.8:
                return numeric_series
        except:
            pass

    # Date detection
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

def filter_and_deduplicate_tables_dicts(tables: List[Dict], min_cols: int, min_rows: int) -> List[Dict]:
    """
    Filter and deduplicate tables based on content, size, and row overlap.
    """
    filtered = [t for t in tables if len(t["table"]) >= min_rows and len(t["table"].columns) >= min_cols]
    
    unique_tables = []
    seen_ranges = []
    for t in filtered:
        is_dup = False
        current_range = (t["start_row"], t["end_row"], t["start_col"], t["end_col"])
        for ut, ut_range in unique_tables:
            # Check for row overlap
            if (max(current_range[0], ut_range[0]) < min(current_range[1], ut_range[1]) and
                max(current_range[2], ut_range[2]) < min(current_range[3], ut_range[3])):
                if tables_are_similar(t["table"], ut["table"], similarity_threshold=0.9):
                    is_dup = True
                    break
        if not is_dup:
            unique_tables.append((t, current_range))
    return [t for t, _ in unique_tables]

def tables_are_similar(table1: pd.DataFrame, table2: pd.DataFrame, similarity_threshold: float = 0.9) -> bool:
    """
    Check if two tables are similar based on content, ignoring header differences.
    """
    if table1.shape != table2.shape:
        return False

    # Compare content, ignoring headers
    sample_size = min(5, len(table1))
    try:
        content_matches = 0
        total_cells = sample_size * len(table1.columns)
        for i in range(sample_size):
            for j in range(len(table1.columns)):
                val1 = table1.iloc[i, j]
                val2 = table2.iloc[i, j]
                if pd.isna(val1) and pd.isna(val2) or str(val1).strip() == str(val2).strip():
                    content_matches += 1
        content_similarity = content_matches / total_cells
        return content_similarity >= similarity_threshold
    except:
        return False

def get_table_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract detailed table metadata for frontend integration.
    """
    if df.empty:
        return {}

    # Ensure 'nan' strings are replaced with None in sample_data
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