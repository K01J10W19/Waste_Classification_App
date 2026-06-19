import React from "react";

export default function PredictionResult({ prediction }) {
  const pct = (prediction.confidence * 100).toFixed(1);
  return (
    <div className="bg-white rounded-xl shadow p-6 space-y-2">
      <h2 className="text-xl font-semibold text-gray-800">Classification Result</h2>
      <p className="text-4xl font-bold text-green-600">{prediction.waste_label}</p>
      <p className="text-gray-500">Confidence: {pct}%</p>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div className="bg-green-500 h-2 rounded-full" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
