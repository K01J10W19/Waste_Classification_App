import axios from "axios";

const api = axios.create({ baseURL: "/api" });

export const classifyWaste = (imageFile) => {
  const form = new FormData();
  form.append("file", imageFile);
  return api.post("/classify", form);
};

export const estimateCarbon = (payload) => api.post("/carbon-estimate", payload);

export const getRecommendations = (wasteLabel) =>
  api.get("/recommendation", { params: { waste_label: wasteLabel } });
