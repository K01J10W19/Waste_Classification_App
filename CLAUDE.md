# CLAUDE.md

This file provides guidance to Claude Code when working on this repository.

## Project Overview

**Title:** Deep Learning Waste Classification and Carbon Impact Estimation Web Application
(Academic title: *Developing a Deep Learning-Based Waste Classification and Carbon Reduction Support Application*)

**Context:** Final Year Project (FYP) — B.Sc. (Hons) Computer Science Specialism in Artificial Intelligence, Asia Pacific University of Technology and Innovation.

**Goal:** Build a web application that helps users dispose of waste properly while understanding its environmental impact. Users upload a waste image, the system classifies it with a CNN, estimates carbon impact via a real-time carbon emission API, and generates disposal/recycling recommendations.

**Alignment:** SDG 11 (Sustainable Cities), SDG 12 (Responsible Consumption), SDG 13 (Climate Action).

---

## System Workflow

```
User uploads waste image
  → Image Preprocessing
  → CNN Waste Classification Model
  → Predicted Waste Type (+ confidence score)
  → User enters estimated weight
  → User selects geographic location
  → Call Carbon Emission API (Climatiq)
  → Receive Carbon Emission Result
  → Generate Recycling Recommendation (rule-based)
  → Display Complete Result on Web Application
```

---

## Modules

### Module 01 — AI Waste Classification (the core AI contribution)
- **Objective:** Train a CNN to classify waste images.
- **Input:** Waste image uploaded by user.
- **Output classes:** Plastic, Paper, Glass, Metal, Cardboard, Organic, (+ other classes depending on dataset — minimum 5: Plastic, Paper, Metal, Organic, Glass).
- **Dataset:** [Recyclable and Household Waste Classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification) — 15,000 images, 30 categories, mapped down to major classes for the model output.
- **Approach:** Transfer learning using **ResNet-50** (justified in IR Chapter 4 conclusion) or MobileNet as a lighter alternative. Use pooling + dropout for generalization.
- **Image format:** Resize to 256x256 px (per IR scope), normalize before inference.
- **Tasks:** dataset preprocessing, image augmentation, training, validation, model export (`.h5` or `.pt`).
- **Evaluation metrics:** Accuracy, Precision, Recall, F1 Score, Confusion Matrix, Accuracy/Loss curves.
- **Explainability:** Integrate SHAP or LIME to produce heatmaps showing which visual features (texture, shape, color) drove the classification — improves user trust ("black box" problem mitigation).

### Module 02 — Carbon Impact Estimation (NOT a regression model)
- **Decision:** No ML regression model. Use a real-time external **Carbon Emission API** instead (Climatiq API, primary choice; Carbon Interface as fallback/alternative).
- **Why:** Provides real-time, IPCC-aligned emission factors without needing to train/maintain a separate regression model. Keeps the AI contribution focused on Module 01 (CNN classification).
- **Workflow:**
  1. CNN predicts material type (e.g., Plastic).
  2. User provides estimated weight (e.g., 0.03 kg) — since exact weighing from an image isn't possible, support a **category average weight method** (e.g., empty plastic bottle ≈ 0.02kg) as a smart default/suggestion, with the user able to override.
  3. User selects geographic location (e.g., Malaysia).
  4. Backend maps the CNN's predicted label to the Climatiq API's standard activity ID (label mapping layer).
  5. Backend calls Climatiq API asynchronously with: material type, weight, location.
  6. API returns carbon emission value (e.g., 0.16 kg CO₂e).
  7. **Multi-path data:** Retrieve CO₂ values for multiple disposal methods in one flow — recycling, incineration, landfilling — to support comparison and ranking.
- **Display:** Show the carbon emission result to the user, plus a comparison across disposal methods.

### Module 03 — Recommendation / Decision Support System
- **Approach:** Rule-based logic. No AI model required here for v1 (an LLM-based dynamic advice layer using Gemini/OpenAI API is a stretch goal — see "Future Enhancements" below, not required for core deliverable).
- **Logic:**
  - **Automated Path Ranking:** Automatically rank the 3 lowest-emission disposal paths for the identified waste type (don't require manual disposal-method selection from the user — reduces friction, validated by 91.7% user approval in IR survey).
  - **Recommendation cards:** Simple, actionable text, e.g., for Plastic Bottle → "Put into recycling bin", "Reuse if possible", "Avoid single-use plastics", "Reduce plastic consumption".
- **Example output:**
  ```
  Prediction: Plastic Bottle
  Estimated Carbon Impact: 0.16 kg CO2e
  Recommendation:
   - Put into recycling bin
   - Reuse if possible
   - Avoid single-use plastics
   - Reduce plastic consumption
  ```

### Module 04 — Web Application
- **Backend:** FastAPI (preferred over Flask — needed for async Climatiq calls, auto-generated OpenAPI docs, Pydantic validation).
- **Frontend:** React + Vite + Tailwind CSS. Vite is used for its dev server and HMR (hot module replacement), which is well suited to iterating quickly on the real-time comparison charts and dashboard components (per IR literature review, section 2.4.6).
- **Database (if required):** SQLite for lightweight server-side storage. Note: the IR's literature review also describes a **client-side Local Storage (Web Storage API)** approach for "Personal Carbon Storage" — i.e., storing scan history and carbon metrics in the browser without a backend DB, for privacy-by-design and zero server setup. Default to this approach for the personal history/dashboard feature unless told otherwise; use SQLite only if persistent multi-device/admin-level data is needed.
- **Website flow:** Home Page → Upload Waste Image → Run CNN Prediction → Display Waste Category + confidence → Ask User Weight → Ask User Location → Call Carbon API → Display Carbon Impact → Display Recycling Recommendation.
- **Dashboard features (from literature review, Module 04 contribution):**
  - Real-time horizontal bar charts comparing CO₂ emissions across recycling/landfill/incineration.
  - Actionable Recommendation Cards.
  - Personal Carbon Storage (Local Storage-based history tracking).

---

## System Architecture

```
Frontend (React + Vite + Tailwind CSS)
    │  HTTP requests
    ▼
FastAPI Backend
    │
    ├──► CNN Model (TensorFlow/Keras, ResNet-50 transfer learning)
    │        → Waste Type Prediction + Confidence
    │
    ├──► Carbon API Service (Climatiq)
    │        → Carbon Emission Result (multi-path: recycle/incinerate/landfill)
    │
    └──► Recommendation Engine (rule-based)
             → Final structured result (type + carbon + advice)
```

This follows an **MVC-inspired structure**:
- Controller layer → `backend/app/api/routes/`
- Model/business logic layer → `backend/app/ml/` + `backend/app/services/`
- View layer → `frontend/`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python (backend, ML), JavaScript (frontend) |
| ML Framework | TensorFlow/Keras (preferred) — PyTorch acceptable alternative |
| CNN Architecture | ResNet-50 (transfer learning), MobileNet as lighter alternative |
| Image Processing | OpenCV, Pillow |
| Backend Framework | FastAPI |
| Frontend | React, Vite, Tailwind CSS |
| Database (optional) | SQLite (server-side) / Web Storage API (client-side, preferred default) |
| Carbon Emission API | Climatiq API (primary), Carbon Interface API (alternative) |
| Explainable AI | SHAP / LIME |
| IDE | VS Code |
| Training environment | Google Colab (GPU/T4) or local with NVIDIA 30-series GPU |
| OS | Development: Windows 11. Deployment: Linux (e.g., AWS) |

---

## Project Folder Structure

```
waste-classification-app/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI entry point
│   │   ├── config.py                  # Env vars, settings
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── classify.py        # POST /classify endpoint
│   │   │   │   ├── carbon.py          # POST /carbon-estimate endpoint
│   │   │   │   └── recommend.py       # GET /recommendation endpoint
│   │   │   └── deps.py                # Shared dependencies (e.g. model loader)
│   │   │
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── model_loader.py        # Loads trained model once at startup
│   │   │   ├── preprocess.py          # Image resize/normalize for inference
│   │   │   └── predict.py             # Runs inference, returns label + confidence
│   │   │
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── carbon_service.py      # Climatiq API client + label mapping
│   │   │   └── recommendation_service.py  # Rule-based advice logic
│   │   │
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── responses.py           # Pydantic models (request/response shapes)
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── logger.py
│   │
│   ├── models/                        # Trained model artifacts
│   │   └── waste_classifier.h5
│   │
│   ├── tests/
│   │   ├── test_classify.py
│   │   ├── test_carbon.py
│   │   └── test_recommend.py
│   │
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
│
├── ml-training/                       # Separate lifecycle from deployed backend
│   ├── data/
│   │   ├── raw/                       # Downloaded Kaggle dataset
│   │   ├── processed/                 # After preprocessing/augmentation
│   │   └── splits/                    # train/val/test
│   │
│   ├── notebooks/
│   │   └── exploration.ipynb
│   │
│   ├── src/
│   │   ├── data_preprocessing.py
│   │   ├── augmentation.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── export_model.py
│   │
│   └── outputs/
│       ├── confusion_matrix.png
│       ├── training_history.png
│       └── classification_report.txt
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── public/
│   │   └── assets/
│   └── src/
│       ├── main.jsx                   # React entry point
│       ├── App.jsx                    # Root component / routing
│       ├── index.css                  # Tailwind directives
│       ├── api/
│       │   └── client.js              # Axios/fetch wrapper for backend calls
│       ├── components/
│       │   ├── ImageUploader.jsx      # Upload + preview
│       │   ├── PredictionResult.jsx   # Waste type + confidence display
│       │   ├── WeightLocationForm.jsx # Weight + location inputs
│       │   ├── CarbonChart.jsx        # Comparison bar chart (recharts/chart.js)
│       │   └── RecommendationCard.jsx # Disposal advice cards
│       ├── pages/
│       │   └── Dashboard.jsx          # Orchestrates the full upload→result flow
│       ├── hooks/
│       │   └── useWasteClassifier.js  # Custom hook wrapping the API call sequence
│       └── utils/
│           └── localStorage.js        # Personal Carbon Storage helper
│
├── docs/                              # Supporting docs for FYP report/appendix
│   ├── architecture.md
│   └── api-documentation.md
│
├── docker-compose.yml
└── README.md
```

---

## Build Phases (work through in order; don't skip ahead without confirming the previous phase is functioning)

1. **Project Scaffolding** — folder structure, environment, dependencies.
2. **Data Pipeline** — download Kaggle dataset, preprocess, augment, split train/val/test.
3. **CNN Model** — build, train, evaluate (ResNet-50/MobileNet transfer learning).
4. **Backend API** — FastAPI app, model inference endpoint, error handling.
5. **Carbon Estimation** — Climatiq API integration, label→activity mapping, weight estimation logic.
6. **Recommendation Engine** — rule-based disposal advice.
7. **Frontend** — upload UI, results dashboard, comparison charts.
8. **Testing** — unit tests per module, integration tests for the full pipeline.
9. **Deployment** — Dockerize, deployment guide (target: Linux/AWS per IR).

---

## Deliverables Checklist

- [ ] Upload a waste image
- [ ] Automatically classify the waste using a CNN model
- [ ] Display prediction confidence
- [ ] Allow user to enter estimated weight
- [ ] Allow user to select location
- [ ] Call a real-time Carbon Emission API
- [ ] Display estimated carbon footprint
- [ ] Display recycling/disposal recommendations
- [ ] Clean, responsive web interface
- [ ] Confusion matrix, accuracy/loss curves, classification report saved to `ml-training/outputs/`
- [ ] Unit + integration tests
- [ ] Dockerized deployment

---

## Explicit Scope Boundaries (from the IR — do not implement these)

- **No** real-time video stream classification — static image upload only.
- **No** industrial/hazardous/medical waste classification — household & recyclable materials only.
- **No** physical hardware/smart bin development — software only.
- **No** regression model for carbon estimation — API-based only.

## Future Enhancements (explicitly out of scope for the core build, mention only if asked)

- Edge AI deployment for low-latency/offline inference.
- Federated Learning for users who want fully local/private data handling.
- IoT-enabled smart bin integration for closed-loop disposal validation.
- LLM-based dynamic advice generation (Gemini/OpenAI API) replacing static rule-based recommendation text.

---

## Coding Conventions

- Follow MVC-inspired separation: routes (controllers) contain no business logic; logic lives in `services/` and `ml/`.
- All API responses should use Pydantic schemas defined in `schemas/`.
- Wrap all external API calls (Climatiq) and model inference in try/except with clear error messages returned via FastAPI's `HTTPException`.
- Keep `ml-training/` fully decoupled from `backend/` — the backend only ever consumes the exported model artifact, never the training code.
- Use environment variables (`.env`, loaded via `config.py`) for API keys — never hardcode the Climatiq API key.
- Write docstrings for every function in `services/` and `ml/` — this project doubles as FYP documentation evidence.
