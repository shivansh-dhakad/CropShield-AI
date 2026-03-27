# 🌿 CropGuard AI — Plant Disease Detection System

A production-ready Flask web app for detecting crop diseases using a two-stage deep learning pipeline. Upload a leaf image → the system identifies the crop → then diagnoses the disease — with full results available in both English and Hindi.

---

## Project Structure

```
project/
├── app.py                     # Flask app — routes, API, startup
├── requirements.txt
├── models/                    # All trained model files go here
│   ├── crop_classification.keras
│   ├── crop_classification.json
│   ├── potato.keras
│   ├── potato.json
│   ├── tomato.keras
│   ├── tomato.json
│   ├── strawberry.keras
│   ├── strawberry.json
│   ├── grapes.keras
│   ├── grapes.json
│   ├── banana.keras
│   ├── banana.json
│   ├── mango.keras
│   └── mango.json
├── static/
│   ├── css/style.css          # Full production UI styles
│   └── js/app.js              # All client-side logic
├── templates/
│   └── index.html             # Single-page app template
└── utils/
    ├── __init__.py
    ├── ml_pipeline.py         # Model loading, caching, prediction pipeline
    └── disease_data.py        # Disease info database + EN/Hindi translations
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the datasets

Download the datasets for all 6 crops from Kaggle and store them on your device. Each crop folder should contain subfolders per disease class.

> **Note:** A free Kaggle account is required to download. Sign up at [kaggle.com](https://www.kaggle.com) if you don't have one.

| Crop | Dataset Link |
|---|---|
| 🍅 Tomato | https://www.kaggle.com/datasets/ashishmotwani/tomato |
| 🍌 Banana | https://www.kaggle.com/datasets/shifatearman/bananalsd |
| 🍓 Strawberry | https://www.kaggle.com/datasets/caozhihao/strawberry-disease-data |
| 🍇 Grapes | https://www.kaggle.com/datasets/zienabesam/grape-plant-from-plant-village-dataset |
| 🥭 Mango | https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset |
| 🍠 Potato | https://www.kaggle.com/datasets/betulbny/potato |

### 3. Train the models

Run the two training files in order:

```bash
# Step 1 — Train the crop classifier (identifies which crop it is)
# Open and run: crop_classification.ipynb

# Step 2 — Train all 6 disease classifiers (one per crop)
python crop_disease_classification.py
```

This will generate the `.keras` model files. Copy all of them along with their corresponding `.json` files into the `models/` directory:

```
models/crop_classification.keras   ← output of crop_classification.ipynb
models/crop_classification.json
models/potato.keras                ← output of crop_disease_classification.py
models/potato.json
models/tomato.keras
models/tomato.json
models/strawberry.keras
models/strawberry.json
models/grapes.keras
models/grapes.json
models/banana.keras
models/banana.json
models/mango.keras
models/mango.json
```

> **Important:** The crop classifier must be named exactly `crop_classification.keras` (underscore, no spaces). If your notebook saved it as `crop classification.keras`, rename it before copying.

### 4. Start the server

```bash
cd project/
python app.py
```

Open: http://localhost:5000

On startup, check the console for this line to confirm all models loaded:

```
Crop classifier: ✅ | Disease models: ['potato', 'tomato', 'strawberry', 'grapes', 'banana', 'mango']
```

---

## How It Works

The system uses a two-stage pipeline:

```
Upload leaf image
        ↓
Stage 1 — Crop Classifier
Identifies: Potato / Tomato / Strawberry / Grape / Banana / Mango
        ↓
Stage 2 — Disease Classifier (crop-specific model)
Identifies the exact disease for that crop
        ↓
Result: Disease name, symptoms, causes, treatment recommendation, severity
```

Keeping the two stages separate means each disease model only focuses on its own crop, which improves accuracy compared to a single large model classifying everything at once.

---

## API Reference

### POST /predict

**Input (multipart/form-data):**
```
image: <image file>  (PNG, JPG, JPEG, WEBP — max 16MB)
```

**Input (JSON — camera capture):**
```json
{ "image_b64": "data:image/jpeg;base64,..." }
```

**Output — supported crop:**
```json
{
  "supported": true,
  "crop": "Tomato",
  "crop_confidence": 94.3,
  "disease": "Tomato___Early_blight",
  "disease_confidence": 87.1,
  "display_name": "Early Blight",
  "symptoms": "Concentric ring spots...",
  "causes": "Fungus Alternaria solani...",
  "recommendation": "Apply chlorothalonil...",
  "severity": "moderate"
}
```

**Output — unsupported crop:**
```json
{
  "supported": false,
  "crop": "wheat",
  "crop_confidence": 88.2,
  "disease": null,
  "display_name": null,
  "symptoms": null,
  "causes": null,
  "recommendation": null,
  "severity": null
}
```

### GET /health

```json
{
  "status": "ok",
  "crop_classifier": true,
  "disease_models_loaded": ["potato", "tomato", "strawberry", "grapes", "banana", "mango"]
}
```

---

## Model Class Name Expectations

The ML pipeline uses these class name arrays. **The order must match the training order exactly** (alphabetical by default when using `image_dataset_from_directory`).

**Crop classifier:** `["banana", "grapes", "mango", "potato", "strawberry", "tomato"]`

**Disease models:**

| Crop | Classes |
|---|---|
| `potato` | `Early_blight`, `Late_blight`, `healthy` |
| `tomato` | `Bacterial_spot`, `Early_blight`, `Late_blight`, `Leaf_Mold`, `Septoria_leaf_spot`, `Spider_mites`, `Yellow_Leaf_Curl_Virus`, `mosaic_virus`, `healthy` |
| `strawberry` | `Leaf_scorch`, `healthy` |
| `grapes` | `Black_rot`, `Esca`, `Leaf_blight`, `healthy` |
| `banana` | `Cordana`, `Pestalotiopsis`, `Sigatoka`, `healthy` |
| `mango` | `Anthracnose`, `Bacterial_Canker`, `Gall_Midge`, `Powdery_Mildew`, `Sooty_Mould`, `healthy` |

> If your dataset folders are in a different order, update the `_class_names` dictionary in `utils/ml_pipeline.py` to match.

---

## Features

- **Two-stage AI pipeline** — crop classification followed by disease classification
- **6 supported crops** — Potato, Tomato, Strawberry, Grape, Banana, Mango
- **28 disease classes** across all crops including healthy states
- **Detailed results** — symptoms, causes, treatment recommendation, severity level
- **Hindi language support** — full UI and results switchable to Hindi with one click
- **Camera support** — capture directly from device camera via WebRTC
- **Drag & drop upload** — or click to browse
- **No model reloading** — all models loaded once at startup and cached in memory

---

## Key Architecture Decisions

| Concern | Solution |
|---|---|
| Model reload per request | `load_all_models()` called once at startup, cached in module globals |
| Camera images | Sent as `multipart/form-data` blob — same path as file upload |
| Language switching | Translation JSON served via `/translations`, applied via `data-key` attributes |
| Error handling | Real errors raised and returned as JSON — no silent fallbacks or mock data |
| Security | File extension validation, 16 MB size limit, no path traversal |
