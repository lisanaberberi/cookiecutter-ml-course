import gradio as gr
import pickle
import mlflow
import pandas as pd
import os
from taxi_ride.models.deploy import Ride, prepare_features

from taxi_ride.data.preprocess_data import  get_project_paths

MODEL_NAME = "random-forest-taxi-model" # Name of the model in MLflow Model Registry

# --- Load DictVectorizer ---
paths = get_project_paths()
data_path = paths["PROCESSED_DATA_DIR"]

DV_PATH = os.path.join(data_path, 'dv.pkl')
with open(DV_PATH, "rb") as f:
    dv = pickle.load(f)

# --- Load MLflow model ---
MODEL_URI = f"models:/{MODEL_NAME}/1"  # adjust if needed
model = mlflow.pyfunc.load_model(MODEL_URI)

# --- Prediction function for Gradio ---
def predict_duration(PULocationID: str, DOLocationID: str, trip_distance: float):
    ride = Ride(
        PULocationID=PULocationID,
        DOLocationID=DOLocationID,
        trip_distance=trip_distance
    )
    X = prepare_features(ride)
    pred = model.predict(X)
    return round(float(pred[0]), 2)

# --- Create Gradio interface ---
iface = gr.Interface(
    fn=predict_duration,
    inputs=[
        gr.Textbox(label="Pickup Location ID"),
        gr.Textbox(label="Dropoff Location ID"),
        gr.Number(label="Trip Distance")
    ],
    outputs=gr.Number(label="Predicted Duration (minutes)"),
    title="Taxi Trip Duration Prediction",
    description="Enter pickup & dropoff location IDs and trip distance to predict trip duration."
)

# --- Launch Gradio app ---
iface.launch(share=True)  # `share=True` allows a public link
