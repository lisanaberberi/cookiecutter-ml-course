import os
import pickle
import click
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


import toml
import os

def find_project_root(start_path=None):
    """
    Walks up from start_path to find the project root,
    defined as a folder containing either .git or pyproject.toml
    """
    if start_path is None:
        start_path = os.path.abspath(os.path.dirname(__file__))

    current = start_path
    while True:
        if os.path.exists(os.path.join(current, ".git")) or \
           os.path.exists(os.path.join(current, "pyproject.toml")):
            return current
        parent = os.path.abspath(os.path.join(current, ".."))
        if parent == current:  # reached filesystem root
            raise FileNotFoundError("Project root not found (no .git or pyproject.toml)")
        current = parent


def get_project_paths():
    """
    Returns absolute paths for RAW_DATA_DIR and PROCESSED_DATA_DIR
    under the project root.
    """
    repo_root = find_project_root()
    raw_data_dir = os.path.join(repo_root, "data/raw")
    processed_dir = os.path.join(repo_root, "data/processed")
    models_dir = os.path.join(repo_root, "models")

    # Ensure directories exist
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True) 

    return {"RAW_DATA_DIR": raw_data_dir, "PROCESSED_DATA_DIR": processed_dir, "MODELS_DIR": models_dir}



def dump_pickle(obj, filename: str):
    """Save object to pickle file."""
    with open(filename, "wb") as f_out:
        pickle.dump(obj, f_out)


def load_pickle(filename: str):
    """Load object from pickle file."""
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    
def load_parquet(path: str) -> pd.DataFrame:
    """
    Load parquet file from local path or URL.
    
    Args:
        path: Local file path or URL to parquet file
        
    Returns:
        DataFrame with loaded data
    """
    if path.startswith(('http://', 'https://')):
        print(f"Loading data from URL: {path}")
    else:
        print(f"Loading data from local file: {path}")
    
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: str):
    """
    Save a DataFrame to a parquet file.
    """
    #print(f"Saving data to local file: {path}")
    df.to_parquet(path, index=False)



def read_dataframe(path: str, dataset: str = "green"):
    """
    Read and preprocess taxi trip dataframe.
    
    Args:
        path: File path or URL to parquet file
        dataset: Type of taxi dataset ('green' or 'yellow')
        
    Returns:
        Preprocessed DataFrame
    """
    df = load_parquet(path)
    
    # Handle different datetime column names for green vs yellow taxi
    if dataset == "green":
        pickup_col = 'lpep_pickup_datetime'
        dropoff_col = 'lpep_dropoff_datetime'
    else:  # yellow
        pickup_col = 'tpep_pickup_datetime'
        dropoff_col = 'tpep_dropoff_datetime'
    
    # Calculate duration
    df['duration'] = df[dropoff_col] - df[pickup_col]
    df['duration'] = df['duration'].apply(lambda td: td.total_seconds() / 60)
    
    # Filter outliers
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    # Convert categorical columns to string
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    """
    Transform dataframe into feature matrix.
    
    Args:
        df: Input dataframe
        dv: DictVectorizer instance
        fit_dv: Whether to fit the vectorizer (True for training data)
        
    Returns:
        Tuple of (feature matrix, fitted vectorizer)
    """
    df['PU_DO'] = df['PULocationID'] + "_" + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    
    return X, dv


def get_data_path(raw_data_path: str, dataset: str, year_month: str) -> str:
    """
    Construct data path - handle both local paths and URLs.
    
    Args:
        raw_data_path: Base path (local directory or base URL)
        dataset: Dataset type ('green' or 'yellow')
        year_month: Year and month in format 'YYYY-MM'
        
    Returns:
        Full path to data file
    """
    filename = f"{dataset}_tripdata_{year_month}.parquet"

   
    if raw_data_path.startswith(('http://', 'https://')):
        # It's a URL - append filename
        return f"{raw_data_path.rstrip('/')}/{filename}"
    else:
        # It's a local path - use os.path.join
        return os.path.join(raw_data_path, filename)


def run_preprocessing(
    raw_data_path: str, 
    dest_path: str, 
    dataset: str = "green",
    train_month: str = "2023-10",
    val_month: str = "2023-11",
    test_month: str = "2023-12"
):
    """
    Run full preprocessing pipeline.
    
    Args:
        raw_data_path: Base path to data (local directory or URL)
        dest_path: Output directory for processed data
        dataset: Dataset type ('green' or 'yellow')
        train_month: Training data month
        val_month: Validation data month
        test_month: Test data month
    """
    # Construct data paths
    train_path = get_data_path(raw_data_path, dataset, train_month)
    val_path = get_data_path(raw_data_path, dataset, val_month)
    test_path = get_data_path(raw_data_path, dataset, test_month)
    
    # Load data
    print("\nLoading training data...")
    df_train = read_dataframe(train_path, dataset)
    print(f"Training data shape: {df_train.shape}")
    
    print("\nLoading validation data...")
    df_val = read_dataframe(val_path, dataset)
    print(f"Validation data shape: {df_val.shape}")
    
    print("\nLoading test data...")
    df_test = read_dataframe(test_path, dataset)
    print(f"Test data shape: {df_test.shape}")

  
    # Extract target
    target = "duration"
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values
    
    # Create and fit DictVectorizer
    print("\nPreprocessing features...")
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv)
    X_test, _ = preprocess(df_test, dv)
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Validation features shape: {X_val.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    # Save processed data
    print(f"\nSaving processed data to {dest_path}...")
    os.makedirs(dest_path, exist_ok=True)
    
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))
    
    print("Done!")


@click.command()
@click.option(
    "--raw_data_path", 
    required=True, 
    help="Directory with raw parquet files or base URL (e.g., https://d37ci6vzurychx.cloudfront.net/trip-data)"
)
@click.option(
    "--dest_path", 
    required=True, 
    help="Where to save processed data."
)
@click.option(
    "--dataset", 
    default="green", 
    help="Dataset color: green or yellow taxi"
)
@click.option(
    "--train_month",
    default="2023-10",
    help="Training data month (YYYY-MM format)"
)
@click.option(
    "--val_month",
    default="2023-11",
    help="Validation data month (YYYY-MM format)"
)
@click.option(
    "--test_month",
    default="2023-12",
    help="Test data month (YYYY-MM format)"
)
def main(
    raw_data_path: str, 
    dest_path: str, 
    dataset: str,
    train_month: str,
    val_month: str,
    test_month: str
):
    """Preprocess taxi trip data for machine learning."""
    run_preprocessing(
        raw_data_path, 
        dest_path, 
        dataset,
        train_month,
        val_month,
        test_month
    )


if __name__ == "__main__":
    main()

# Usage examples:
# Load from URL (as in your notebook)
# python src/taxi_ride/data/preprocess_data.py \
#   --raw_data_path https://d37ci6vzurychx.cloudfront.net/trip-data \
#   --dest_path data/raw \
#   --dataset green

# # Load from local directory
# python preprocess_data.py \
#   --raw_data_path ./data \
#   --dest_path ./output \
#   --dataset yellow