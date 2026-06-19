import React from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";

const METHOD_COLORS = { recycling: "#22c55e", incineration: "#f97316", landfill: "#6b7280", composting: "#84cc16" };

export default function CarbonChart({ data }) {
  return (
    <div className="bg-white rounded-xl shadow p-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">CO₂ Emissions by Disposal Method</h2>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} layout="vertical" margin={{ left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" unit=" kg" />
          <YAxis type="category" dataKey="method" width={100} />
          <Tooltip formatter={(v) => `${v.toFixed(4)} kg CO₂e`} />
          <Bar dataKey="co2e_kg">
            {data.map((entry) => (
              <Cell key={entry.method} fill={METHOD_COLORS[entry.method] ?? "#60a5fa"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
