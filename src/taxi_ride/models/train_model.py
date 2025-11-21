import os
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from taxi_ride.data.preprocess_data import load_pickle, get_project_paths
from datetime import datetime 

# Experiment
EXPERIMENT_NAME = "random-forest-experiments"

# Hyperparameters to log
RF_PARAMS = ["max_depth", "n_estimators", "min_samples_split",
             "min_samples_leaf", "random_state"]


# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog(log_input_examples=False, log_model_signatures=False)

# Default data path
paths = get_project_paths()
DEFAULT_DATA_PATH = paths["PROCESSED_DATA_DIR"]

# Set run name as date and time
run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def train_and_log_model(data_path=DEFAULT_DATA_PATH, params=None):
    """Train RandomForest with given params and log metrics."""

    #print("data-path",DEFAULT_DATA_PATH)

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))


    with mlflow.start_run(run_name=run_name) as run:
        # Convert parameters to integers
        new_params = {p: int(params[p]) for p in RF_PARAMS}

        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        # Predictions
        val_pred = rf.predict(X_val)
        test_pred = rf.predict(X_test)

        # Metrics
        val_rmse = root_mean_squared_error(y_val, val_pred)
        test_rmse = root_mean_squared_error(y_test, test_pred)
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)

        # Log metrics
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("val_r2", val_r2)
        mlflow.log_metric("test_r2", test_r2)

        print(f"Run ID: {run.info.run_id}")
        print(f"Run name: {run_name}")
        print(f"Logged model with params {new_params}")
        print(f"Validation R2: {val_r2:.4f}, Test R2: {test_r2:.4f}")
    
    return run.info.run_id, val_rmse, test_rmse, val_r2, test_r2

@click.command()
@click.option("--data_path", default=DEFAULT_DATA_PATH, help="Path to processed data")
@click.option("--max_depth", default=10, type=int)
@click.option("--n_estimators", default=50, type=int)
@click.option("--min_samples_split", default=2, type=int)
@click.option("--min_samples_leaf", default=1, type=int)
@click.option("--random_state", default=42, type=int)
def run_train(data_path, max_depth, n_estimators, min_samples_split, min_samples_leaf, random_state):
    params = {
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "random_state": random_state
    }
    train_and_log_model(data_path, params)

if __name__ == "__main__":
    run_train()
