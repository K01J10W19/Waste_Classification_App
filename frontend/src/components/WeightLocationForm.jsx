import React, { useState } from "react";

const LOCATIONS = ["MY", "US", "GB", "AU", "SG", "DE", "IN"];

export default function WeightLocationForm({ onSubmit, loading }) {
  const [weight, setWeight] = useState("");
  const [location, setLocation] = useState("MY");

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({ weight_kg: parseFloat(weight), location });
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-xl shadow p-6 space-y-4">
      <h2 className="text-xl font-semibold text-gray-800">Estimate Carbon Impact</h2>
      <div className="flex gap-4">
        <div className="flex-1">
          <label className="block text-sm text-gray-600 mb-1">Weight (kg)</label>
          <input
            type="number"
            step="0.001"
            min="0.001"
            required
            value={weight}
            onChange={(e) => setWeight(e.target.value)}
            placeholder="e.g. 0.02"
            className="w-full border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-400"
          />
        </div>
        <div className="flex-1">
          <label className="block text-sm text-gray-600 mb-1">Location</label>
          <select
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            className="w-full border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-400"
          >
            {LOCATIONS.map((l) => <option key={l} value={l}>{l}</option>)}
          </select>
        </div>
      </div>
      <button
        type="submit"
        disabled={loading}
        className="w-full bg-green-600 text-white rounded-lg py-2 font-medium hover:bg-green-700 disabled:opacity-50"
      >
        {loading ? "Calculating..." : "Calculate Carbon Impact"}
      </button>
    </form>
  );
}
