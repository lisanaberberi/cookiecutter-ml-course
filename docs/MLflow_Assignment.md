# MLflow Model Training Exercises

## Exercise 1: Hyperparameter Tuning Analysis

### Objective
Explore how different hyperparameters affect model performance by training Random Forest models with various configurations and analyzing the results in MLflow.

### Background
You have been provided with a baseline model trained using the following command:

```bash
python src/taxi_ride/models/train_model.py --data_path ./data/processed/ \
    --max_depth 12 --n_estimators 150 --min_samples_split 2 \
    --min_samples_leaf 1 --random_state 42
```

### Tasks

#### Part A: Vary `n_estimators`
Train models with different values of `n_estimators` while keeping other parameters constant:

1. **Model 1**: `n_estimators=100`
   ```bash
   python src/taxi_ride/models/train_model.py --data_path ./data/processed/ \
       --max_depth 12 --n_estimators 50 --min_samples_split 2 \
       --min_samples_leaf 1 --random_state 42
   ```

2. **Model 2**: `n_estimators=150`
   ```bash
   python src/taxi_ride/models/train_model.py --data_path ./data/processed/ \
       --max_depth 12 --n_estimators 100 --min_samples_split 2 \
       --min_samples_leaf 1 --random_state 42
   ```
 

#### Part B: Vary `max_depth`
Train models with different values of `max_depth` while keeping other parameters constant:

1. **Model 3**: `max_depth=5`
   ```bash
   python src/taxi_ride/models/train_model.py --data_path ./data/processed/ \
       --max_depth 5 --n_estimators 150 --min_samples_split 2 \
       --min_samples_leaf 1 --random_state 42
   ```

2. **Model 4**: `max_depth=10`
   ```bash
   python src/taxi_ride/models/train_model.py --data_path ./data/processed/ \
       --max_depth 10 --n_estimators 150 --min_samples_split 2 \
       --min_samples_leaf 1 --random_state 42
   ```

3. **Model 5**: `max_depth=12` (baseline)


### Questions to Answer

1. **Record the metrics**: For each of the 5 models trained above, record the following metrics from MLflow:
   - `val_rmse`
   - `test_rmse`
   - `val_r2`
   - `test_r2`

2. **Analysis Questions**:
   
   a) **Effect of `n_estimators`**:
      - Does the test RMSE improve consistently as you increase `n_estimators`?
   
   b) **Effect of `max_depth`**:
      - Which `max_depth` value gives the best test R² score?
   
   c) **Best Model**:
      - How much improvement (in percentage) does the best model show compared to the baseline?

3. **MLflow UI Exploration**:
   - Use the MLflow UI (http://127.0.0.1:5000) to compare all models
   - Create a screenshot showing the comparison of all 5 models sorted by `test_r2`
   - Which visualization in MLflow helps you understand the hyperparameter effects best?

### Deliverables

Submit a table in the following format:

| Model | n_estimators | max_depth | val_rmse | test_rmse | val_r2 | test_r2 |
|-------|--------------|-----------|----------|-----------|--------|---------|
| 1     | 50           | 12        | ?        | ?         | ?      | ?       |
| 2     | 100          | 12        | ?        | ?         | ?      | ?       |
| ...   | ...          | ...       | ...      | ...       | ...    | ...     |


---

## Exercise 2: Feature Engineering Impact Analysis

### Objective
Evaluate how adding additional numerical features affects model performance by modifying the preprocessing pipeline and comparing R² scores.

### Background
Currently, the model uses only `trip_distance` as a numerical feature:

```python
numerical = ['trip_distance']
```

Your task is to modify the feature set to include additional numerical features and measure the impact on model performance.

### Current Setup

The baseline model is trained with:
- **Features**: `numerical = ['trip_distance']`
- **Hyperparameters**: 
  ```bash
  --max_depth 12 --n_estimators 150 --min_samples_split 2 \
  --min_samples_leaf 1 --random_state 42
  ```

### Task

#### Step 1: Train Baseline Model
First, train a baseline model with the current feature set to establish a benchmark.

```bash
python src/taxi_ride/models/train_model.py --data_path ./data/processed/ \
    --max_depth 12 --n_estimators 150 --min_samples_split 2 \
    --min_samples_leaf 1 --random_state 42
```

Record the metrics (especially `test_r2` and `test_rmse`).

#### Step 2: Modify the Feature Set

Locate the preprocessing script (likely in `src/taxi_ride/data/preprocess_data.py` or similar) and change:

**FROM:**
```python
numerical = ['trip_distance']
```

**TO:**
```python
numerical = ['fare_amount', 'total_amount', 'trip_distance']
```

#### Step 3: Reprocess the Data

After modifying the feature set, you'll need to rerun the data preprocessing pipeline to regenerate the training, validation, and test sets with the new features.

```bash
python src/taxi_ride/data/preprocess_data.py
```

#### Step 4: Train New Model with Extended Features

Train a new model using the same hyperparameters but with the expanded feature set:

```bash
python src/taxi_ride/models/train_model.py --data_path ./data/processed/ \
    --max_depth 12 --n_estimators 150 --min_samples_split 2 \
    --min_samples_leaf 1 --random_state 42
```

Record the metrics for this new model.

### Questions to Answer

1. **Metrics Comparison**:
   
   | Configuration | Features Used | test_r2 | test_rmse | val_r2 | val_rmse |
   |---------------|---------------|---------|-----------|--------|----------|
   | Baseline      | trip_distance only | ? | ? | ? | ? |
   | Extended      | fare_amount, total_amount, trip_distance | ? | ? | ? | ? |

2. **Performance Analysis**:
   
   a) **Did the R² score improve?**
      - Yes / No
      - By how much? (Calculate the absolute difference and percentage improvement)
      - Formula: `Improvement (%) = ((New R² - Old R²) / Old R²) × 100`
   
   b) **Did the RMSE improve?**
      - Yes / No
      - By how much? (Calculate the absolute difference and percentage improvement)
   
   c) **Validation vs Test Performance**:
      - Is there a bigger improvement in validation or test metrics?
      - What might this tell you about the model's generalization?

3. **Feature Importance** (Optional Advanced Task):
   
   If you have access to feature importance from the Random Forest model:
   - Which of the three numerical features is most important?
   - Does `trip_distance` remain the most important feature?
   - How do `fare_amount` and `total_amount` rank?


### Deliverables

Submit:

1. **Metrics Table**: Completed table showing baseline vs extended feature set performance

2. **MLflow Screenshots**: 
   - Screenshot comparing the two runs in MLflow UI
   - Highlight the difference in R² scores


---

## Submission Guidelines

For both exercises:

1. **Document all commands** you ran
2. **Record all metrics** from MLflow (you can export them from the UI or query programmatically)
3. **Provide screenshots** of the MLflow UI showing your experiments
4. **Format your submission** as a PDF or Markdown document

## Evaluation Criteria

- **Completeness**: All models trained and metrics recorded
- **Accuracy**: Correct calculations and metric interpretations
- **MLflow Usage**: Proper use of MLflow for tracking and comparison
- **Critical Thinking**: Insightful observations about hyperparameters and features


Good luck! 🚀