import { useState } from "react";
import { classifyWaste, estimateCarbon as apiEstimate, getRecommendations } from "../api/client.js";
import { saveScan } from "../utils/localStorage.js";

const initialState = { prediction: null, carbonData: null, recommendations: null, classifying: false, estimating: false, error: null };

export default function useWasteClassifier() {
  const [state, setState] = useState(initialState);

  const classify = async (file) => {
    setState((s) => ({ ...s, classifying: true, error: null }));
    try {
      const { data } = await classifyWaste(file);
      setState((s) => ({ ...s, prediction: data, classifying: false }));
    } catch (e) {
      setState((s) => ({ ...s, classifying: false, error: e.message }));
    }
  };

  const estimateCarbon = async ({ weight_kg, location }) => {
    setState((s) => ({ ...s, estimating: true, error: null }));
    try {
      const [carbonRes, recRes] = await Promise.all([
        apiEstimate({ waste_label: state.prediction.waste_label, weight_kg, location }),
        getRecommendations(state.prediction.waste_label),
      ]);
      saveScan({ ...state.prediction, ...carbonRes.data, timestamp: Date.now() });
      setState((s) => ({ ...s, carbonData: carbonRes.data, recommendations: recRes.data, estimating: false }));
    } catch (e) {
      setState((s) => ({ ...s, estimating: false, error: e.message }));
    }
  };

  const reset = () => setState(initialState);

  return { state, classify, estimateCarbon, reset };
}
