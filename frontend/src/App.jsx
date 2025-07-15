import React from "react";
import FileUploader from "./components/FileUploader";

function App() {
  return (
    <div className="min-h-screen bg-white p-10 max-w-5xl mx-auto">
      <h1 className="text-3xl font-bold mb-6 text-gray-800">
        RevealIQ — Instant Excel Insights
      </h1>
      <FileUploader />
    </div>
  );
}

export default App;
