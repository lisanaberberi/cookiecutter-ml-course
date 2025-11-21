from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import mlflow
import pandas as pd
import uvicorn
import os
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from taxi_ride.data.preprocess_data import  get_project_paths

mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "random-forest-taxi-model" # Name of the model in MLflow Model Registry

# --- Load DictVectorizer ---

paths = get_project_paths()
data_path = paths["PROCESSED_DATA_DIR"]

DV_PATH = os.path.join(data_path, 'dv.pkl')
with open(DV_PATH, "rb") as f:
    dv = pickle.load(f)

# --- Load MLflow model ---
MODEL_URI = f"models:/{MODEL_NAME}/1"  # adjust version if needed
model = mlflow.pyfunc.load_model(MODEL_URI)

# --- Define FastAPI app ---
app = FastAPI(title="Taxi Trip Duration Prediction API")

# --- Input schema ---
class Ride(BaseModel):
    PULocationID: str
    DOLocationID: str
    trip_distance: float

# --- Preprocessing function ---
def prepare_features(ride: Ride):
    """Transform user input into vectorized feature matrix."""
    PU_DO = f"{ride.PULocationID}_{ride.DOLocationID}"
    X = dv.transform([{"PU_DO": PU_DO, "trip_distance": ride.trip_distance}])
    return X

# --- Prediction endpoint ---
@app.post("/predict")
def predict_endpoint(ride: Ride):
    try:
        X = prepare_features(ride)
        pred = model.predict(X)
        return {"duration": float(pred[0])}
    except Exception as e:
        return {"error": str(e)}

# --- Optional: simple homepage ---
@app.get("/")
def home():
    return {"message": "Send POST requests to /predict with PULocationID, DOLocationID, trip_distance."}


@app.get("/predict_form", response_class=HTMLResponse)
def predict_form():
    return """
    <html>
        <body>
            <h2>Taxi Trip Duration Prediction</h2>
            <form action="/predict_form" method="post">
                Pickup Location ID: <input type="text" name="PULocationID"><br>
                Dropoff Location ID: <input type="text" name="DOLocationID"><br>
                Trip Distance: <input type="number" step="0.1" name="trip_distance"><br>
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """

@app.post("/predict_form", response_class=HTMLResponse)
def predict_form_post(
    PULocationID: str = Form(...),
    DOLocationID: str = Form(...),
    trip_distance: float = Form(...)
):
    # Create Ride object
    ride = Ride(
        PULocationID=PULocationID,
        DOLocationID=DOLocationID,
        trip_distance=trip_distance
    )
    # Predict
    X = prepare_features(ride)
    pred = model.predict(X)
    return f"<h3>Predicted duration: {float(pred[0]):.2f} minutes</h3>"

# --- Run app ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
