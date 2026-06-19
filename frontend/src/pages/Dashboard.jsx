import React, { useState } from "react";
import ImageUploader from "../components/ImageUploader.jsx";
import PredictionResult from "../components/PredictionResult.jsx";
import WeightLocationForm from "../components/WeightLocationForm.jsx";
import CarbonChart from "../components/CarbonChart.jsx";
import RecommendationCard from "../components/RecommendationCard.jsx";
import useWasteClassifier from "../hooks/useWasteClassifier.js";

export default function Dashboard() {
  const { state, classify, estimateCarbon, reset } = useWasteClassifier();

  return (
    <main className="max-w-3xl mx-auto px-4 py-10 space-y-8">
      <h1 className="text-3xl font-bold text-green-700 text-center">
        Waste Classification & Carbon Impact
      </h1>

      <ImageUploader onUpload={classify} loading={state.classifying} />

      {state.prediction && (
        <>
          <PredictionResult prediction={state.prediction} />
          <WeightLocationForm onSubmit={estimateCarbon} loading={state.estimating} />
        </>
      )}

      {state.carbonData && (
        <>
          <CarbonChart data={state.carbonData.estimates} />
          <RecommendationCard
            wasteLabel={state.prediction.waste_label}
            recommendations={state.recommendations}
          />
        </>
      )}

      {(state.prediction || state.carbonData) && (
        <button onClick={reset} className="text-sm text-gray-400 underline block mx-auto">
          Start over
        </button>
      )}
    </main>
  );
}
