import React, { useState } from "react";
import axios from "axios";
import {
  Sparkles,
  Copy,
  Upload,
  Download,
  Loader2,
  AlertTriangle,
  FileCheck
} from "lucide-react";

const FileUploader = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [columns, setColumns] = useState([]);
  const [previewRows, setPreviewRows] = useState([]);
  const [selectedTableIndex, setSelectedTableIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);
  const [isAnalyzed, setIsAnalyzed] = useState(false);
  const [overview, setOverview] = useState(null);
  const [overviewLoading, setOverviewLoading] = useState(false);


  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError(null);
    setResult(null);
    setPreviewRows([]);
    setColumns([]);
    setSelectedTableIndex(0);
    setCopied(false);
    setIsAnalyzed(false);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file first.");
      return;
    }
    setLoading(true);
    setError(null);
    setIsAnalyzed(false);
    setOverview(null);
    setOverviewLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await axios.post("http://localhost:8000/upload/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const data = res.data;
      setResult(data);

      if (data.overview) {
        setOverview(data.overview);
        setOverviewLoading(false);
      }
      
      if (data.tables && data.tables.length > 0) {
        const table = data.tables[0];
        setSelectedTableIndex(0);
        setColumns(table.column_names || []);
        setPreviewRows(table.preview || []);
      } else if (data.non_tabular_sections > 0) {
        setError("No tabular data found, but non-tabular content detected. Review the data below.");
      }
    } catch (err) {
      setError(err.response?.data?.error || "Upload failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };
  
  

  const handleCleanAndAnalyze = () => {
    setIsAnalyzed(true);
  };

  const handleTableChange = (e) => {
    const index = parseInt(e.target.value);
    if (!result?.tables || !result.tables[index]) {
      setSelectedTableIndex(0);
      setColumns([]);
      setPreviewRows([]);
      return;
    }
    setSelectedTableIndex(index);
    const table = result.tables[index];
    setColumns(table.column_names || []);
    setPreviewRows(table.preview || []);
    setCopied(false);
  };

  const handleCopy = () => {
    if (!columns.length || !previewRows.length) return;
    const csv = columns.join(",") + "\n" + previewRows.map(row =>
      columns.map(col => {
        const value = row[col];
        return value === null || value === undefined || String(value).toLowerCase() === "nan" 
          ? "" 
          : JSON.stringify(value).replace(/^"|"$/g, '');
      }).join(",")
    ).join("\n");
    navigator.clipboard.writeText(csv);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const handleExport = () => {
    if (!columns.length || !previewRows.length) return;
    const csv = columns.join(",") + "\n" + previewRows.map(row =>
      columns.map(col => {
        const value = row[col];
        return value === null || value === undefined || String(value).toLowerCase() === "nan" 
          ? "" 
          : JSON.stringify(value).replace(/^"|"$/g, '');
      }).join(",")
    ).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${result.tables[selectedTableIndex]?.sheet_name || "table"}-table${selectedTableIndex + 1}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const Metric = ({ label, value }) => (
  <div className="bg-white rounded-lg border border-slate-200 p-3 flex flex-col items-center justify-center">
    <span className="text-slate-500 text-xs">{label}</span>
    value={`${overview?.overall_completeness?.toFixed?.(1) ?? 0}%`}
    <span className="text-slate-800 text-base font-semibold">{value}</span>
  </div>
);


  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100">
      <header className="bg-white border-b border-slate-200/60 sticky top-0 z-30 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-6 flex justify-between items-center">
          <div className="flex gap-3 items-center">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center shadow-lg">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold gradient-text">RevealIQ</h1>
              <p className="text-sm text-slate-600">Transform your data into insights instantly</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-600">
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
            <span>System Online</span>
          </div>
        </div>
        <div className="h-1 w-full bg-gradient-to-r from-blue-500 via-purple-500 to-emerald-400 animate-slideIn"></div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-12 space-y-10 animate-fadeIn">
        <div className="bg-white rounded-xl p-6 shadow-md border border-slate-200">
          <h2 className="text-xl font-semibold mb-4">Upload Excel or CSV File</h2>
          <input
            type="file"
            accept=".xlsx,.xls,.csv"
            onChange={handleFileChange}
            className="mb-4"
          />
          <button
            onClick={handleUpload}
            className="btn-primary flex items-center gap-2"
            disabled={loading}
          >
            {loading ? <Loader2 className="animate-spin w-4 h-4" /> : <Upload className="w-4 h-4" />}
            {loading ? "Uploading..." : "Upload"}
          </button>
          {error && (
            <p className="text-red-600 mt-4 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" /> {error}
            </p>
          )}
        </div>

{result && (
  <div className="bg-white rounded-xl p-6 shadow-md border border-slate-200 space-y-6">

    {/* 🌟 Data Overview Section */}
    {overview && (
      <div className="bg-gradient-to-br from-blue-50 via-white to-purple-50 rounded-xl border border-blue-100 p-6 mb-6 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-800 mb-3 flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-blue-600" />
          Data Overview
        </h2>

        {overviewLoading ? (
          <p className="text-slate-500 text-sm">Calculating overview...</p>
        ) : (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-slate-700 text-sm">
              <Metric label="Tables Processed" value={overview.tables_processed} />
              <Metric label="Total Rows" value={overview.rows_total} />
              <Metric label="Total Columns" value={overview.columns_total} />
              <Metric label="Missing Values Fixed" value={overview.missing_values_fixed} />
              <Metric label="Duplicate Rows Removed" value={overview.duplicate_rows_removed} />
              <Metric label="Date Columns Standardized" value={overview.date_columns_standardized} />
              <Metric
                label="Overall Completeness"
                value={`${overview.overall_completeness.toFixed(1)}%`}
              />
            </div>

            {/* 🟩 Animated Completeness Bar */}
            <div className="mt-4">
              <p className="text-xs text-slate-500 mb-1">Overall Completeness</p>
              <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                <div
                  className="h-2 bg-gradient-to-r from-green-400 via-emerald-500 to-teal-500 rounded-full transition-all duration-700 ease-out"
                  style={{ width: `${overview.overall_completeness}%` }}
                ></div>
              </div>
            </div>
          </>
        )}
      </div>
    )}

    {/* 📊 Table Preview Header */}
    <div className="flex items-center justify-between">
      <div>
        <h2 className="text-xl font-semibold">Table Preview</h2>
        <p className="text-slate-500 text-sm">
          Showing:{" "}
          <span className="font-medium">
            {result.tables[selectedTableIndex]?.sheet_name
              ? `${result.tables[selectedTableIndex].sheet_name} - Table ${
                  result.tables[selectedTableIndex].table_id ||
                  selectedTableIndex + 1
                }`
              : `Table ${selectedTableIndex + 1}`}
            {result.tables[selectedTableIndex]?.is_tabular
              ? " (Tabular)"
              : " (Non-Tabular)"}
          </span>
        </p>
      </div>

      <div className="flex items-center gap-3">
        {result.tables.length > 1 && (
          <select
            value={selectedTableIndex}
            onChange={handleTableChange}
            className="border border-slate-300 rounded-md px-3 py-1 text-sm"
          >
            {result.tables.map((t, idx) => (
              <option key={idx} value={idx}>
                {t.sheet_name
                  ? `${t.sheet_name} - Table ${t.table_id || idx + 1} ${
                      t.is_tabular ? "(Tabular)" : "(Non-Tabular)"
                    }`
                  : `Table ${idx + 1} ${
                      t.is_tabular ? "(Tabular)" : "(Non-Tabular)"
                    }`}
              </option>
            ))}
          </select>
        )}
        {!isAnalyzed && (
          <button
            onClick={handleCleanAndAnalyze}
            className="btn-primary flex items-center gap-2"
          >
            <FileCheck className="w-4 h-4" />
            Clean and Analyze
          </button>
        )}
      </div>
    </div>

    {/* ⚠️ Non-tabular warning */}
    {result.non_tabular_sections > 0 && (
      <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4 text-sm text-yellow-800 flex items-start gap-2">
        <AlertTriangle className="w-5 h-5" />
        <div>
          <p>
            Found {result.non_tabular_sections} non-tabular section(s). These
            may contain notes, comments, or unstructured data.
          </p>
          <p>
            Review the data below and consider manual processing for
            non-tabular content.
          </p>
        </div>
      </div>
    )}

    {/* 🧮 Table Content */}
    {columns.length > 0 && previewRows.length > 0 ? (
      <div className="overflow-x-auto border border-slate-200 rounded-lg">
        <table className="min-w-full table-auto text-sm text-slate-800">
          <thead className="bg-slate-100 border-b border-slate-200">
            <tr>
              {columns.map((col, i) => (
                <th key={i} className="px-3 py-2 text-left">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {previewRows.map((row, i) => (
              <tr
                key={i}
                className="border-b border-slate-100 hover:bg-slate-50"
              >
                {columns.map((col, j) => (
                  <td key={j} className="px-3 py-2">
                    {row[col] === null ||
                    row[col] === undefined ||
                    String(row[col]).toLowerCase() === "nan"
                      ? ""
                      : row[col]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    ) : (
      <p className="text-slate-500 text-sm">
        No preview available for this{" "}
        {result.tables[selectedTableIndex]?.is_tabular
          ? "table"
          : "section"}
        .
      </p>
    )}

    {/* 📈 Table Stats */}
    {isAnalyzed && (
      <div className="mt-4 text-sm text-slate-700">
        <p>
          Rows in table:{" "}
          {result.tables[selectedTableIndex]?.rows ?? "N/A"}
        </p>
        <p>
          Columns in table:{" "}
          {result.tables[selectedTableIndex]?.columns ?? "N/A"}
        </p>
        <p>
          Confidence Score:{" "}
          {(
            result.tables[selectedTableIndex]?.confidence_score * 100
          ).toFixed(2)}
          %
        </p>
        <p>
          Header Row:{" "}
          {result.tables[selectedTableIndex]?.header_row !== null
            ? result.tables[selectedTableIndex]?.header_row + 1
            : "None detected"}
        </p>

        {result.tables[selectedTableIndex]?.stats?.missing_data && (
          <>
            <p className="mt-2 font-semibold">
              Missing Values per Column:
            </p>
            <ul className="list-disc list-inside max-h-40 overflow-y-auto">
              {Object.entries(
                result.tables[selectedTableIndex].stats.missing_data
              ).map(([col, val]) => (
                <li key={col}>
                  {col}: {val}
                </li>
              ))}
            </ul>
          </>
        )}

        {result.tables[selectedTableIndex]?.stats?.data_quality && (
          <>
            <p className="mt-2 font-semibold">Data Quality:</p>
            <p>
              Completeness:{" "}
              {result.tables[
                selectedTableIndex
              ].stats.data_quality.completeness.toFixed(2)}
              %
            </p>
            <p>
              Duplicate Rows:{" "}
              {
                result.tables[selectedTableIndex].stats.data_quality
                  .duplicate_rows
              }
            </p>
          </>
        )}
      </div>
    )}

    {/* 💾 Copy & Export */}
    {columns.length > 0 && previewRows.length > 0 && (
      <div className="flex gap-3">
        <button
          onClick={handleCopy}
          className="btn-secondary flex items-center gap-1"
        >
          <Copy className="w-4 h-4" />
          {copied ? "Copied!" : "Copy CSV"}
        </button>
        <button
          onClick={handleExport}
          className="btn-secondary flex items-center gap-1"
        >
          <Download className="w-4 h-4" />
          Export CSV
        </button>
      </div>
    )}
  </div>
)}

        
      </main>
    </div>
  );
};

export default FileUploader;