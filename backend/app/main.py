from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from io import BytesIO
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from app.utils import detect_and_clean_tables, get_table_info

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Excel/CSV Sheet Processor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    # Check for non-tabular patterns (e.g., long text blocks)
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns[:5]:
        non_null = df[col].dropna()
        if len(non_null) > 0 and non_null.str.len().mean() > 100:
            analysis["potential_issues"].append(f"Column {col} contains long text, possibly notes")
            analysis["is_likely_tabular"] = False
            analysis["recommendations"].append("Consider extracting key-value pairs from text")

    return analysis

@app.post("/upload/")
async def upload_excel(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing file: {file.filename}")
        contents = await file.read()

        file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
        all_sheets = {}

        if file_extension in ['xlsx', 'xls']:
            try:
                logger.info("Reading as Excel file...")
                all_sheets = pd.read_excel(
                    BytesIO(contents),
                    sheet_name=None,
                    header=None,
                    keep_default_na=False,
                    na_values=['']
                )
                logger.info(f"Successfully loaded {len(all_sheets)} Excel sheets: {list(all_sheets.keys())}")
            except Exception as e:
                logger.error(f"Excel read failed: {e}")
                raise HTTPException(status_code=400, detail=f"Could not read Excel file: {str(e)}")

        elif file_extension == 'csv':
            try:
                logger.info("Reading as CSV file...")
                csv_df = pd.read_csv(BytesIO(contents), header=None, keep_default_na=False, na_values=[''])
                all_sheets = {'CSV': csv_df}
                logger.info("Successfully loaded CSV file")
            except Exception as e:
                logger.error(f"CSV read failed: {e}")
                raise HTTPException(status_code=400, detail=f"Could not read CSV file: {str(e)}")
        else:
            try:
                logger.info("Trying to read as Excel...")
                all_sheets = pd.read_excel(BytesIO(contents), sheet_name=None, header=None)
                logger.info(f"Successfully loaded as Excel: {list(all_sheets.keys())}")
            except Exception as e1:
                logger.info(f"Excel read failed: {e1}, trying CSV...")
                try:
                    csv_df = pd.read_csv(BytesIO(contents), header=None)
                    all_sheets = {'CSV': csv_df}
                    logger.info("Successfully loaded as CSV")
                except Exception as e2:
                    logger.error(f"Both Excel and CSV reads failed: {e1}, {e2}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Could not read file as Excel or CSV: {str(e1)}"
                    )

        if not all_sheets:
            raise HTTPException(status_code=400, detail="No sheets found in the uploaded file")

        processed_sheets = []
        total_tables_found = 0
        non_tabular_sections = 0

        for sheet_name, df in all_sheets.items():
            logger.info(f"Processing sheet: {sheet_name} (shape: {df.shape})")

            # Clean sheet first
            df_cleaned = df.dropna(how='all').dropna(axis=1, how='all')
            if df_cleaned.empty:
                logger.info(f"Skipping empty or garbage sheet: {sheet_name}")
                processed_sheets.append({
                    "sheet_name": sheet_name,
                    "table_id": 0,
                    "table_name": f"{sheet_name} - Empty",
                    "rows": 0,
                    "columns": 0,
                    "column_names": [],
                    "preview": [],
                    "stats": {},
                    "sheet_analysis": analyze_sheet_structure(df_cleaned, sheet_name),
                    "start_row": None,
                    "end_row": None,
                    "start_col": None,
                    "end_col": None,
                    "confidence_score": 0.0,
                    "is_tabular": False
                })
                continue

            sheet_analysis = analyze_sheet_structure(df_cleaned, sheet_name)
            if not sheet_analysis["has_data"]:
                processed_sheets.append({
                    "sheet_name": sheet_name,
                    "table_id": 0,
                    "table_name": f"{sheet_name} - No Data",
                    "rows": 0,
                    "columns": 0,
                    "column_names": [],
                    "preview": [],
                    "stats": {},
                    "sheet_analysis": sheet_analysis,
                    "start_row": None,
                    "end_row": None,
                    "start_col": None,
                    "end_col": None,
                    "confidence_score": 0.0,
                    "is_tabular": False
                })
                continue

            tables = detect_and_clean_tables(df_cleaned)
            logger.info(f"Found {len(tables)} tables in sheet {sheet_name}")

            for i, table_dict in enumerate(tables):
                table = table_dict["table"]
                if table.empty:
                    continue
                try:
                    table_info = get_table_info(table)
                    preview_data = safe_json_convert(table, max_rows=10)
                    data_types = {col: str(dtype) for col, dtype in table.dtypes.items()}
                    missing_data = convert_numpy_types(table.isnull().sum().to_dict())
                    non_null_counts = convert_numpy_types(table.notna().sum().to_dict())

                    processed_table = {
                        "sheet_name": sheet_name,
                        "table_id": i + 1,
                        "table_name": f"{sheet_name} - Table {i + 1}",
                        "rows": int(len(table)),
                        "columns": int(len(table.columns)),
                        "column_names": table.columns.tolist(),
                        "preview": preview_data,
                        "stats": {
                            "data_types": data_types,
                            "missing_data": missing_data,
                            "non_null_counts": non_null_counts,
                            "data_quality": convert_numpy_types(table_info.get("data_quality", {})),
                            "memory_usage": convert_numpy_types(table_info.get("memory_usage", 0))
                        },
                        "sheet_analysis": sheet_analysis,
                        "start_row": table_dict.get("start_row"),
                        "end_row": table_dict.get("end_row"),
                        "start_col": table_dict.get("start_col"),
                        "end_col": table_dict.get("end_col"),
                        "confidence_score": table_dict.get("confidence_score"),
                        "is_tabular": table_dict.get("is_tabular"),
                        "header_row": table_dict.get("header_row")
                    }

                    processed_sheets.append(processed_table)
                    total_tables_found += 1
                    if not table_dict.get("is_tabular"):
                        non_tabular_sections += 1

                except Exception as e:
                    logger.error(f"Error processing table {i+1} in sheet {sheet_name}: {e}")
                    continue

        if total_tables_found == 0:
            return JSONResponse(
                content=convert_numpy_types({
                    "success": False,
                    "error": "No usable tables found in the file",
                    "details": "The file may contain only text, images, or unstructured data",
                    "sheets_analyzed": len(all_sheets),
                    "tables_found": 0,
                    "non_tabular_sections": non_tabular_sections
                }),
                status_code=400
            )

        response = {
            "success": True,
            "message": f"Successfully processed {total_tables_found} table(s) from {len(all_sheets)} sheet(s)",
            "sheets_analyzed": len(all_sheets),
            "tables_found": total_tables_found,
            "non_tabular_sections": non_tabular_sections,
            "tables": processed_sheets,
            "file_info": {
                "filename": file.filename,
                "file_type": file_extension,
                "sheets": list(all_sheets.keys()) if file_extension != 'csv' else []
            }
        }

        if total_tables_found == 1:
            main_table = processed_sheets[0]
            response.update({
                "columns": main_table["column_names"],
                "row_count": main_table["rows"],
                "preview": main_table["preview"],
                "sheet_name": main_table["sheet_name"],
                "is_tabular": main_table["is_tabular"]
            })

        return JSONResponse(content=convert_numpy_types(response))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing file: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": f"Processing failed: {str(e)}",
                "details": "An unexpected error occurred while processing the file"
            },
            status_code=500
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Excel/CSV processor is running"}

@app.get("/")
async def root():
    return {
        "message": "Excel/CSV Sheet Processor API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload/ (POST)",
            "health": "/health (GET)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)