from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from io import BytesIO
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional

from app.config import settings
from app.schemas import UploadResponse, TableSectionMeta, TableStats, FileInfo
from app.processors.table_processor import TableProcessor
from app.DataCleaner import DataCleaner
from app.schemas import UploadResponse, TableSectionMeta, TableStats, FileInfo, DataOverview

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.APP_NAME, version=settings.VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = TableProcessor(logger=logger)

# ---- Helpers identical to your original (kept as-is) ----
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj) or str(obj).lower() == "nan":
        return None
    else:
        return obj

def safe_json_convert(df: pd.DataFrame, max_rows: int = 10) -> List[Dict[str, Any]]:
    try:
        preview_df = df.head(max_rows).copy()
        for col in preview_df.columns:
            if preview_df[col].dtype == 'datetime64[ns]':
                preview_df[col] = preview_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            elif preview_df[col].dtype == 'object':
                preview_df[col] = preview_df[col].astype(str).replace(r'^(?i)nan$', '', regex=True)
        preview_df = preview_df.replace({np.nan: None, pd.NA: None, 'nan': None, 'NaN': None})
        records = preview_df.to_dict(orient="records")
        return convert_numpy_types(records)
    except Exception as e:
        logger.error(f"Error converting preview data to JSON: {e}")
        return []

def analyze_sheet_structure(df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
    analysis = {
        "sheet_name": sheet_name,
        "original_shape": list(df.shape),
        "has_data": not df.empty,
        "data_density": 0.0,
        "potential_issues": [],
        "recommendations": [],
        "is_likely_tabular": True
    }
    if df.empty:
        analysis["potential_issues"].append("Sheet is completely empty")
        analysis["is_likely_tabular"] = False
        return analysis

    total_cells = df.shape[0] * df.shape[1]
    non_null_cells = int(df.notna().sum().sum())
    analysis["data_density"] = float((non_null_cells / total_cells) * 100) if total_cells > 0 else 0.0

    if analysis["data_density"] < 10:
        analysis["potential_issues"].append("Very sparse data (< 10% filled)")
        analysis["is_likely_tabular"] = False
        analysis["recommendations"].append("May contain non-tabular content (e.g., notes or comments)")

    row_lengths = [int(df.iloc[i].notna().sum()) for i in range(min(10, len(df)))]
    if len(set(row_lengths)) > 3:
        analysis["potential_issues"].append("Irregular row lengths detected")
        analysis["recommendations"].append("Check for multiple tables or unstructured data")

    if df.shape[0] > 20:
        empty_rows = int(df.isnull().all(axis=1).sum())
        if empty_rows > 3:
            analysis["recommendations"].append("Multiple tables or sections may be present")

    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns[:5]:
        non_null = df[col].dropna()
        if len(non_null) > 0 and non_null.str.len().mean() > 100:
            analysis["potential_issues"].append(f"Column {col} contains long text, possibly notes")
            analysis["is_likely_tabular"] = False
            analysis["recommendations"].append("Consider extracting key-value pairs from text")
    return analysis

# ---- Routes ----

@app.post("/upload/", response_model=UploadResponse)
async def upload_excel(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing file: {file.filename}")
        contents = await file.read()
        ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
        all_sheets = {}

        if ext in ['xlsx', 'xls']:
            try:
                all_sheets = pd.read_excel(BytesIO(contents), sheet_name=None, header=None,
                                           keep_default_na=False, na_values=[''])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not read Excel file: {str(e)}")
        elif ext == 'csv':
            try:
                csv_df = pd.read_csv(BytesIO(contents), header=None, keep_default_na=False, na_values=[''])
                all_sheets = {'CSV': csv_df}
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not read CSV file: {str(e)}")
        else:
            # best-effort fallback
            try:
                all_sheets = pd.read_excel(BytesIO(contents), sheet_name=None, header=None)
            except Exception as e1:
                try:
                    csv_df = pd.read_csv(BytesIO(contents), header=None)
                    all_sheets = {'CSV': csv_df}
                except Exception as e2:
                    raise HTTPException(status_code=400, detail=f"Could not read file as Excel or CSV: {str(e1)}")

        if not all_sheets:
            raise HTTPException(status_code=400, detail="No sheets found in the uploaded file")

        processed_tables: List[TableSectionMeta] = []
        total_tables_found = 0
        non_tabular_sections = 0
        
        #Overview accumulators (merged from /data-overview/)
        total_tables = 0
        total_rows = 0
        total_cols = 0
        missing_values_fixed = 0
        date_cols_standardized = 0
        duplicate_rows_removed = 0
        completeness_scores = []

        for sheet_name, df in all_sheets.items():
            df_cleaned = df.dropna(how='all').dropna(axis=1, how='all')
            sheet_analysis = analyze_sheet_structure(df_cleaned, sheet_name)

            if not sheet_analysis["has_data"]:
                continue

            detected = processor.detect_tables(
                df_cleaned,
                min_data_density=settings.MIN_DATA_DENSITY,
                min_cols=settings.MIN_COLS,
                min_rows=settings.MIN_ROWS,
                max_gap=settings.MAX_GAP
            )

            for i, table_dict in enumerate(detected):
                table = table_dict["table"]
                if table.empty:
                    continue
             # ---- Cleaning phase ----
                cleaner = DataCleaner(table)
                cleaner.remove_empty().fill_missing('mean').standardize_dates()
                cleaned_table = cleaner.get_cleaned_df()
                cleaning_log = cleaner.get_summary()
                
                 # ---- Overview metrics (using cleaned table) ----
                total_tables += 1
                total_rows += len(cleaned_table)
                total_cols += len(cleaned_table.columns)

                original_missing = int(table.isna().sum().sum())
                cleaned_missing = int(cleaned_table.isna().sum().sum())
                fixed = max(0, original_missing - cleaned_missing)
                missing_values_fixed += fixed

                # Date columns standardized (heuristic similar to /data-overview/)
                for col in cleaned_table.columns:
                    try:
                        if str(cleaned_table[col].dtype).startswith('datetime64') or (
                            cleaned_table[col].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}$').sum()
                            > len(cleaned_table) * 0.8
                        ):
                            date_cols_standardized += 1
                    except Exception:
                        continue

                duplicate_rows_removed += int(table.duplicated().sum())

                completeness = (
                    cleaned_table.notna().sum().sum()
                    / (len(cleaned_table) * len(cleaned_table.columns))
                    * 100
                ) if len(cleaned_table) > 0 and len(cleaned_table.columns) > 0 else 0.0
                completeness_scores.append(completeness)


                tinfo = processor.get_table_info(table)
                preview = safe_json_convert(table, max_rows=10)

                stats = TableStats(
                    data_types={col: str(dtype) for col, dtype in table.dtypes.items()},
                    missing_data={k: int(v) for k, v in table.isnull().sum().to_dict().items()},
                    non_null_counts={k: int(v) for k, v in table.notna().sum().to_dict().items()},
                    data_quality=tinfo.get("data_quality"),
                    memory_usage=float(tinfo.get("memory_usage", 0))
                )

                processed_tables.append(
                    TableSectionMeta(
                        sheet_name=sheet_name,
                        table_id=i + 1,
                        table_name=f"{sheet_name} - Table {i + 1}",
                        rows=int(len(table)),
                        columns=int(len(table.columns)),
                        column_names=list(table.columns),
                        preview=preview,
                        stats=stats,
                        cleaning_actions=cleaning_log["changes"],  # ✅ new
                        start_row=table_dict.get("start_row"),
                        end_row=table_dict.get("end_row"),
                        start_col=table_dict.get("start_col"),
                        end_col=table_dict.get("end_col"),
                        header_row=table_dict.get("header_row"),
                        confidence_score=float(table_dict.get("confidence_score") or 0.0),
                        is_tabular=bool(table_dict.get("is_tabular"))
                    )
                )

                total_tables_found += 1
                if not table_dict.get("is_tabular"):
                    non_tabular_sections += 1

        if total_tables_found == 0:
            raise HTTPException(status_code=400, detail="No usable tables found in the file")
        
        overall_completeness = float(
            round(np.mean(completeness_scores), 2)
        ) if completeness_scores else 0.0

        file_info = FileInfo(
            filename=file.filename,
            file_type=ext or "unknown",
            sheets=list(all_sheets.keys()) if ext != 'csv' else []
        )

        resp = UploadResponse(
            success=True,
            message=f"Successfully processed {total_tables_found} table(s) from {len(all_sheets)} sheet(s)",
            sheets_analyzed=len(all_sheets),
            tables_found=total_tables_found,
            non_tabular_sections=non_tabular_sections,
            tables=processed_tables,
            file_info=file_info,
            overview=DataOverview(
                tables_processed=total_tables,
                rows_total=total_rows,
                columns_total=total_cols,
                missing_values_fixed=missing_values_fixed,
                date_columns_standardized=date_cols_standardized,
                duplicate_rows_removed=duplicate_rows_removed,
                overall_completeness=overall_completeness,
            )
        )


        if total_tables_found == 1:
            main_table = processed_tables[0]
            resp.columns = main_table.column_names
            resp.row_count = main_table.rows
            resp.preview = main_table.preview
            resp.sheet_name = main_table.sheet_name
            resp.is_tabular = main_table.is_tabular

        # FastAPI will auto-serialize thanks to response_model
        return JSONResponse(content=convert_numpy_types(resp.dict()))


    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    



@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Excel/CSV processor is running"}

@app.get("/")
async def root():
    return {
        "message": settings.APP_NAME,
        "version": settings.VERSION,
        "endpoints": {
            "upload": "/upload/ (POST)",
            "health": "/health (GET)"
        }
    }
