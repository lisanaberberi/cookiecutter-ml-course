import os
import pickle
import click
import mlflow
import pandas as pd

MODEL_NAME = "random-forest-taxi-model" # Name of the model in MLflow Model Registry

mlflow.set_tracking_uri("http://127.0.0.1:5000")

from taxi_ride.data.preprocess_data import load_pickle, get_project_paths

paths = get_project_paths()
data_path = paths["PROCESSED_DATA_DIR"]
models_path = paths["MODELS_DIR"]


@click.command()
@click.option("--data_path", default=data_path)
@click.option("--output_path", default=f"{models_path}/predictions.pkl")
def predict(data_path, output_path):

    print("Loading model from MLflow Model Registry...")
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")

    X_test, _ = load_pickle(os.path.join(data_path, "test.pkl"))

    preds = model.predict(X_test)
    print(f"Generated {len(preds)} predictions.")

    with open(output_path, "wb") as f:
        pickle.dump(preds, f)

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    predict()
