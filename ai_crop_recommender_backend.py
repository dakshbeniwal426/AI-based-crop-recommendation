# ai_crop_recommender_backend.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import uvicorn
import os, sys

# ---------------------------
# Create FastAPI app
# ---------------------------
app = FastAPI(title="AI Crop Recommendation API", version="1.0")

# ---------------------------
# Enable CORS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Serve Frontend (index.html)
# ---------------------------
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    # If static/index.html exists serve it; otherwise return a small message
    index_path = "static/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found. Place index.html in the static/ folder."}

# ---------------------------
# Input Models
# ---------------------------
class RecommendInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    humidity: float
    temperature: float
    rainfall: float

class CropRecommendation(BaseModel):
    crop: str
    score: float

class RecommendOutput(BaseModel):
    recommendations: List[CropRecommendation]

class FertilizerInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    target_crop: str

class FertilizerOutput(BaseModel):
    suggestions: Dict[str, str]

# ---------------------------
# Load CSV and auto-detect columns
# ---------------------------
CSV_FILE = "crop_data.csv"   # change if your file has a different name

if not os.path.exists(CSV_FILE):
    raise RuntimeError(f"❌ {CSV_FILE} not found in {os.getcwd()}. Put your dataset in the project root.")

df = pd.read_csv(CSV_FILE)
print("Loaded CSV. Columns:", list(df.columns))

# 1) detect target column (crop)
candidate_names = ["crop", "Crop", "label", "Label", "target", "Target", "crops"]
target_col = None
for c in candidate_names:
    if c in df.columns:
        target_col = c
        break

if target_col is None:
    # try to find non-numeric columns (likely the crop name column)
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if len(non_numeric) == 1:
        target_col = non_numeric[0]
        print(f"Auto-detected target (non-numeric) column: {target_col}")
    elif len(non_numeric) > 1:
        print("Multiple non-numeric columns found:", non_numeric)
        # fallback to last column
        target_col = df.columns[-1]
        print(f"Falling back to last column as target: {target_col}")
    else:
        # no non-numeric columns — fallback to last column
        target_col = df.columns[-1]
        print(f"No non-numeric columns found. Using last column as target: {target_col}")

print("Using target column:", target_col)

# 2) Prepare label encoder for the target
le = LabelEncoder()
try:
    df[target_col] = le.fit_transform(df[target_col])
except Exception as e:
    print("Error encoding target column:", e)
    raise

# 3) Identify feature columns (flexible matching)
# canonical features we want
feature_candidates = {
    "N": ["N", "n", "nitrogen", "nitro"],
    "P": ["P", "p", "phosphorus", "phos"],
    "K": ["K", "k", "potassium", "potash"],
    "ph": ["ph", "pH", "PH"],
    "humidity": ["humidity", "moisture"],
    "temperature": ["temperature", "temp", "avg_temp", "avg temp", "avg-temp"],
    "rainfall": ["rainfall", "rain", "rain_mm", "rain_mm", "precipitation"]
}

# build a mapping from lower-case col name -> actual
colmap = {c.lower(): c for c in df.columns}

selected_features = {}
missing_features = []
for key, variants in feature_candidates.items():
    found = None
    for v in variants:
        if v.lower() in colmap:
            found = colmap[v.lower()]
            break
    if found:
        selected_features[key] = found

# report
print("Detected feature columns (mapping):")
for k, v in selected_features.items():
    print(f"  {k} -> {v}")

# Check required ones
required = ["N", "P", "K", "ph", "humidity", "temperature", "rainfall"]
for r in required:
    if r not in selected_features:
        missing_features.append(r)

if missing_features:
    # Provide clear message and show available columns
    print("\nERROR: Missing expected feature columns:", missing_features)
    print("Available CSV columns:", list(df.columns))
    print("Please rename your CSV columns or update the mapping in the backend.")
    raise RuntimeError(f"Missing columns required for prediction: {missing_features}")

# 4) Build X and y using the selected feature names (in stable order)
feature_order = [selected_features[k] for k in required]
X = df[feature_order]
y = df[target_col]

print("Training model using features:", feature_order)

# 5) Train-test split and model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Optionally print training accuracy
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Model trained. Train acc: {train_acc:.3f}, Test acc: {test_acc:.3f}")

# ---------------------------
# Endpoints
# ---------------------------
@app.post("/recommend", response_model=RecommendOutput)
def recommend(data: RecommendInput):
    """ Recommend crops using ML model """

    # create a data row mapping to the dataset's column names
    row = {
        selected_features["N"]: data.nitrogen,
        selected_features["P"]: data.phosphorus,
        selected_features["K"]: data.potassium,
        selected_features["ph"]: data.ph,
        selected_features["humidity"]: data.humidity,
        selected_features["temperature"]: data.temperature,
        selected_features["rainfall"]: data.rainfall
    }

    input_data = pd.DataFrame([row], columns=feature_order)

    probs = model.predict_proba(input_data)[0]
    crop_labels = le.inverse_transform(model.classes_)  # model.classes_ are encoded labels

    top_indices = probs.argsort()[-3:][::-1]
    recommendations = [
        CropRecommendation(crop=crop_labels[i], score=round(probs[i] * 100, 2))
        for i in top_indices
    ]
    return RecommendOutput(recommendations=recommendations)

@app.post("/fertilizer", response_model=FertilizerOutput)
def fertilizer(data: FertilizerInput):
    """ Suggest fertilizers based on soil nutrient levels """
    suggestions = {}
    suggestions["Nitrogen"] = (
        "Add Urea or Ammonium Sulfate" if data.nitrogen < 50 else "Nitrogen level is sufficient"
    )
    suggestions["Phosphorus"] = (
        "Add DAP or Rock Phosphate" if data.phosphorus < 40 else "Phosphorus level is sufficient"
    )
    suggestions["Potassium"] = (
        "Add MOP (Muriate of Potash)" if data.potassium < 40 else "Potassium level is sufficient"
    )
    suggestions["General"] = f"Ensure proper irrigation for {data.target_crop}"
    return FertilizerOutput(suggestions=suggestions)

# ---------------------------
# Run backend (when script executed directly)
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # run app by passing the app object (avoids module import issues)
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)