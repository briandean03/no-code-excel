from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class TableStats(BaseModel):
    data_types: Dict[str, str]
    missing_data: Dict[str, int]
    non_null_counts: Dict[str, int]
    data_quality: Optional[Dict[str, Any]] = None
    memory_usage: Optional[float] = None

class TableSectionMeta(BaseModel):
    sheet_name: str
    table_id: int
    table_name: str
    rows: int
    columns: int
    column_names: List[str]
    preview: List[Dict[str, Any]]
    stats: TableStats
    start_row: Optional[int] = None
    end_row: Optional[int] = None
    start_col: Optional[int] = None
    end_col: Optional[int] = None
    header_row: Optional[int] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    is_tabular: bool

class FileInfo(BaseModel):
    filename: str
    file_type: str
    sheets: List[str] = []

class UploadResponse(BaseModel):
    success: bool
    message: str
    sheets_analyzed: int
    tables_found: int
    non_tabular_sections: int
    tables: List[TableSectionMeta]
    file_info: FileInfo
    # optional fields for single-table convenience
    columns: Optional[List[str]] = None
    row_count: Optional[int] = None
    preview: Optional[List[Dict[str, Any]]] = None
    sheet_name: Optional[str] = None
    is_tabular: Optional[bool] = None
