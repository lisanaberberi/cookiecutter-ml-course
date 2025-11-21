import os
import click
import numpy as np
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from taxi_ride.models.predict_model import predict as generate_predictions
from taxi_ride.data.preprocess_data import load_pickle, get_project_paths

paths = get_project_paths()
data_path = paths["PROCESSED_DATA_DIR"]
models_path = paths["MODELS_DIR"]
MODEL_NAME = "rf-taxi-duration"

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("random-forest-experiments")


@click.command()
@click.option("--data_path", default=data_path, help="Path to test data")
@click.option("--predictions_path", default=f"{models_path}/predictions.pkl", help="Path to saved predictions")
@click.option("--generate", is_flag=True, help="Generate predictions first")
def visualize(data_path, predictions_path, generate):
    """
    Visualise and evaluate predictions against actual values from the test dataset.
    """
    
    # Generate predictions first if requested
    if generate:
        print("Generating predictions first...")
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(generate_predictions, [
            '--data_path', data_path,
            '--output_path', predictions_path
        ])
        if result.exit_code != 0:
            print("Error generating predictions:")
            print(result.output)
            return
        print("Predictions generated successfully.\n")
    
    # Load test data
    print("Loading test data...")
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))
    
    # Load predictions
    print(f"Loading predictions from {predictions_path}...")
    import pickle
    with open(predictions_path, "rb") as f:
        preds = pickle.load(f)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Error analysis
    errors = np.abs(y_test - preds)
    percentage_errors = (errors / y_test) * 100
    
    # Print comprehensive report
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    print(f"\n📊 Performance Metrics:")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  MAE (Mean Absolute Error):      {mae:.4f}")
    print(f"  R² Score:                       {r2:.4f}")
    
    print(f"\n🔍 Error Analysis:")
    print(f"  Mean Absolute Error:       {errors.mean():.4f}")
    print(f"  Median Absolute Error:     {np.median(errors):.4f}")
    print(f"  Std Dev of Errors:         {errors.std():.4f}")
    print(f"  95th Percentile Error:     {np.percentile(errors, 95):.4f}")
    print(f"  Mean Percentage Error:     {percentage_errors.mean():.2f}%")
    print(f"  Median Percentage Error:   {np.median(percentage_errors):.2f}%")
    
    print(f"\n📈 Prediction Range:")
    print(f"  Actual:    [{y_test.min():.2f}, {y_test.max():.2f}]")
    print(f"  Predicted: [{preds.min():.2f}, {preds.max():.2f}]")
    print(f"  Mean Actual:    {y_test.mean():.2f}")
    print(f"  Mean Predicted: {preds.mean():.2f}")
    
    # Accuracy buckets
    print(f"\n✅ Prediction Accuracy Distribution:")
    within_5_pct = (percentage_errors <= 5).sum()
    within_10_pct = (percentage_errors <= 10).sum()
    within_20_pct = (percentage_errors <= 20).sum()
    total = len(percentage_errors)
    
    print(f"  Within 5% of actual:   {within_5_pct:5d} ({within_5_pct/total*100:.1f}%)")
    print(f"  Within 10% of actual:  {within_10_pct:5d} ({within_10_pct/total*100:.1f}%)")
    print(f"  Within 20% of actual:  {within_20_pct:5d} ({within_20_pct/total*100:.1f}%)")
    
    # Find worst predictions
    worst_idx = np.argsort(errors)[-10:]
    print(f"\n❌ Top 10 Worst Predictions:")
    print(f"  {'Rank':<6} {'Actual':<10} {'Predicted':<12} {'Error':<10} {'% Error':<10}")
    print(f"  {'-'*58}")
    for i, idx in enumerate(worst_idx[::-1], 1):
        print(f"  {i:<6} {y_test[idx]:<10.2f} {preds[idx]:<12.2f} "
              f"{errors[idx]:<10.2f} {percentage_errors[idx]:<10.2f}%")
    
    # Find best predictions
    best_idx = np.argsort(errors)[:10]
    print(f"\n✨ Top 10 Best Predictions:")
    print(f"  {'Rank':<6} {'Actual':<10} {'Predicted':<12} {'Error':<10} {'% Error':<10}")
    print(f"  {'-'*58}")
    for i, idx in enumerate(best_idx, 1):
        print(f"  {i:<6} {y_test[idx]:<10.2f} {preds[idx]:<12.2f} "
              f"{errors[idx]:<10.2f} {percentage_errors[idx]:<10.2f}%")
    
    print("\n" + "="*60)
    
    # Log metrics to MLflow
    print("\n📝 Logging metrics to MLflow...")
    with mlflow.start_run(run_name="prediction_evaluation"):
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("mean_percentage_error", percentage_errors.mean())
        mlflow.log_metric("median_percentage_error", np.median(percentage_errors))
        mlflow.log_metric("accuracy_within_5pct", within_5_pct/total*100)
        mlflow.log_metric("accuracy_within_10pct", within_10_pct/total*100)
        mlflow.log_metric("accuracy_within_20pct", within_20_pct/total*100)
    
    print("✅ Metrics logged to MLflow successfully!")


if __name__ == "__main__":
    visualize()