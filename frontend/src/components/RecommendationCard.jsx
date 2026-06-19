import React from "react";

export default function RecommendationCard({ wasteLabel, recommendations }) {
  if (!recommendations) return null;
  return (
    <div className="bg-green-50 border border-green-200 rounded-xl shadow p-6 space-y-3">
      <h2 className="text-xl font-semibold text-green-800">
        Disposal Recommendations — {wasteLabel}
      </h2>
      <ul className="space-y-2">
        {recommendations.tips.map((tip, i) => (
          <li key={i} className="flex items-start gap-2 text-green-900">
            <span className="mt-1 text-green-500">✓</span>
            <span>{tip}</span>
          </li>
        ))}
      </ul>
      <p className="text-sm text-gray-500">
        Best method: <strong className="text-green-700">{recommendations.ranked_methods[0]}</strong>
      </p>
    </div>
  );
}
