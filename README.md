# Waste Classification App

Deep Learning-Based Waste Classification and Carbon Reduction Support Application

**FYP — B.Sc. (Hons) Computer Science (AI Specialism), Asia Pacific University**

---

## Project Overview

Users upload a waste image → CNN classifies it → carbon impact is estimated via the Climatiq API → disposal recommendations are displayed.

**Tech Stack:** Python · FastAPI · TensorFlow/Keras (ResNet-50) · React · Vite · Tailwind CSS

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A [Kaggle account](https://www.kaggle.com) with an API token (`kaggle.json`)

### 1. Clone the repository

```bash
git clone <repo-url>
cd waste-classification-app
```

### 2. Set up Kaggle credentials

Place your `kaggle.json` token at `~/.kaggle/kaggle.json`:

```bash
# Windows
mkdir $env:USERPROFILE\.kaggle
copy kaggle.json $env:USERPROFILE\.kaggle\kaggle.json

# macOS / Linux
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download the dataset

The dataset (~920 MB) is not included in the repository. Run these two commands to download and organise it:

```bash
kaggle datasets download \
  -d alistairking/recyclable-and-household-waste-classification \
  -p ml-training/data/raw --unzip

python ml-training/src/data_preprocessing.py
```

This downloads 14,000 images across 30 categories and splits them into `train / val / test` sets (80/10/10) mapped to 6 major waste classes: **plastic, paper, glass, metal, cardboard, organic**.

### 4. Set up the backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # then fill in your API keys
```

### 5. Set up the frontend

```bash
cd frontend
npm install
```

---

## Running the App

```bash
# Backend (from /backend)
uvicorn app.main:app --reload

# Frontend (from /frontend)
npm run dev
```

---

## Build Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Project scaffolding | Done |
| 2 | Data pipeline (download, preprocess, augment) | Done |
| 3 | CNN model training (ResNet-50) | Pending |
| 4 | Backend API (FastAPI, inference endpoint) | Pending |
| 5 | Carbon estimation (Climatiq API) | Pending |
| 6 | Recommendation engine (rule-based) | Pending |
| 7 | Frontend (React dashboard) | Pending |
| 8 | Testing | Pending |
| 9 | Deployment (Docker) | Pending |

---

## Dataset

[Recyclable and Household Waste Classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification) — 14,000 images, 30 categories, mapped to 6 major classes.

**License:** MIT
