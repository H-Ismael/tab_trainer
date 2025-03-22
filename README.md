# fastAPI-based Machine Learning model training api

## Overview
this API, developed using fastAPI, enables the automated training of machine learning models, specifically Support Vector Classification (SVC) and xgboost. By providing a structured interface for dataset ingestion, hyperparameter configuration, and performance evaluation, it facilitates efficient experimentation with different model architectures and optimization strategies.

## Features
- dynamic dataset ingestion via api endpoints
- implementation of SVC and xgboost classifiers
- optional hyperparameter tuning using Optuna
- configurable parameters, including test size and random state
- robust logging for monitoring and debugging

## Installation

Ensure Python (>=3.8) is installed. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

## Running the API

To deploy the API locally, execute:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Train support vector classifier (SVC)
```http
POST /train/svc
```
#### Request Parameters (multipart/form-data):
- `data_file` (file, required): CSV file containing feature vectors and labels
- `label_column` (string, required): Name of the target variable column
- `test_column` (string, optional): Name of a predefined test set column
- `test_size` (float, default: 0.2): proportion of the dataset allocated for validation
- `random_state` (int, default: 42): Seed value for reproducibility
- `use_optuna` (bool, default: False): Enable hyperparameter optimization
- `hyperparams` (string, optional): JSON-formatted string specifying hyperparameters
- `n_trials` (int, default: 10): Number of trials for optuna tuning

#### Sample Response:
```json
{
  "accuracy": 0.92,
  "f1_score": 0.88,
  "confusion_matrix": [[50, 5], [4, 41]]
}
```

### Train xgboost classifier
```http
POST /train/xgboost
```
#### Request Parameters (identical to SVC endpoint)

#### Sample Response:
```json
{
  "accuracy": 0.94,
  "f1_score": 0.90,
  "confusion_matrix": [[52, 3], [2, 43]]
}
```

## Logging & gpu Utilization
- The api includes structured logging to track significant events and errors.
- gpu availability can be checked, but this functionality is currently disabled.

## Future Enhancements
- standardize dataset storage, retrieval, and deletion
- implement early stopping based on performance metrics
- refine evaluation metric selection for improved generalization

## License
this project is distributed under the [BSD License](LICENSE).

