import pandas as pd
import logging
from typing import Optional, IO
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import joblib
import json

logger = logging.getLogger(__name__)

def train_svc(
    data_file: IO,
    label_column: str,
    test_column: Optional[str],
    test_size: float,
    random_state: int,
    use_optuna: bool,
    hyperparams: Optional[str]
):
    logger.info("Starting SVC training")

    # Read data
    try:
        data = pd.read_csv(data_file)
        logger.info("Data loaded successfully")
    except Exception as e:
        logger.exception("Failed to read data file")
        raise e

    # Split data
    if test_column and test_column in data.columns:
        train_data = data[data[test_column] == 0]
        test_data = data[data[test_column] == 1]
        logger.info(f"Using '{test_column}' column for train/test split")
    else:
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state)
        logger.info(f"Splitting data with test_size={test_size} and random_state={random_state}")

    X_train = train_data.drop(columns=[label_column])
    y_train = train_data[label_column]
    X_test = test_data.drop(columns=[label_column])
    y_test = test_data[label_column]

    # SVC does not support GPU acceleration
    logger.info("SVC does not support GPU acceleration. Training on CPU.")

    # Parse hyperparameters
    if hyperparams:
        hyperparams = json.loads(hyperparams)
    else:
        hyperparams = {}

    if use_optuna:
        # Perform hyperparameter optimization with Optuna
        import optuna

        def objective(trial):
            C = trial.suggest_float('C', 1e-5, 1e2, log=True)
            gamma = trial.suggest_float('gamma', 1e-5, 1e1, log=True)
            model = SVC(C=C, gamma=gamma, random_state=random_state)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            f1 = classification_report(y_test, preds, output_dict=True)['weighted avg']['f1-score']
            return f1

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        best_params = study.best_params
        logger.info(f"Best hyperparameters found: {best_params}")
        model = SVC(**best_params, random_state=random_state)
    else:
        model = SVC(**hyperparams, random_state=random_state)
        logger.info(f"Training SVC with hyperparameters: {hyperparams}")

    # Train model
    model.fit(X_train, y_train)
    logger.info("Model training completed")

    # Evaluate model
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    logger.info("Model evaluation completed")
    logger.info(f"Classification Report: \n{classification_report(y_test, preds)}")

    # Save model
    model_filename = "svc_model.pkl"
    joblib.dump(model, model_filename)
    logger.info(f"Model saved to {model_filename}")

    # Return metrics
    metrics = {
        'accuracy': report['accuracy'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score'],
        'precision': report['weighted avg']['precision']
    }

    return metrics

def train_xgboost(
    data_file: IO,
    label_column: str,
    test_column: Optional[str],
    test_size: float,
    random_state: int,
    use_optuna: bool,
    hyperparams: Optional[str],
    gpu_available: bool
):
    logger.info("Starting XGBoost training")

    # Read data
    try:
        data = pd.read_csv(data_file)
        logger.info("Data loaded successfully")
    except Exception as e:
        logger.exception("Failed to read data file")
        raise e

    # Split data
    if test_column and test_column in data.columns:
        train_data = data[data[test_column] == 0]
        test_data = data[data[test_column] == 1]
        logger.info(f"Using '{test_column}' column for train/test split")
    else:
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state)
        logger.info(f"Splitting data with test_size={test_size} and random_state={random_state}")

    X_train = train_data.drop(columns=[label_column])
    y_train = train_data[label_column]
    X_test = test_data.drop(columns=[label_column])
    y_test = test_data[label_column]

    # Use GPU if available
    if gpu_available:
        tree_method = 'gpu_hist'
        logger.info("Using GPU for XGBoost training")
    else:
        tree_method = 'hist'
        logger.info("GPU not available. Using CPU for XGBoost training")

    # Parse hyperparameters
    if hyperparams:
        hyperparams = json.loads(hyperparams)
    else:
        hyperparams = {}

    if use_optuna:
        # Perform hyperparameter optimization with Optuna
        import optuna
        import xgboost as xgb

        def objective(trial):
            param = {
                'tree_method': tree_method,
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1.0, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'random_state': random_state
            }

            model = xgb.XGBClassifier(**param)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            f1 = classification_report(y_test, preds, output_dict=True)['weighted avg']['f1-score']
            return f1

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        best_params = study.best_params
        best_params['tree_method'] = tree_method
        best_params['random_state'] = random_state
        logger.info(f"Best hyperparameters found: {best_params}")
        model = xgb.XGBClassifier(**best_params)
    else:
        import xgboost as xgb
        hyperparams['tree_method'] = tree_method
        hyperparams['random_state'] = random_state
        model = xgb.XGBClassifier(**hyperparams)
        logger.info(f"Training XGBoost with hyperparameters: {hyperparams}")

    # Train model
    model.fit(X_train, y_train)
    logger.info("Model training completed")

    # Evaluate model
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    logger.info("Model evaluation completed")
    logger.info(f"Classification Report: \n{classification_report(y_test, preds)}")

    # Save model
    model_filename = "xgboost_model.pkl"
    joblib.dump(model, model_filename)
    logger.info(f"Model saved to {model_filename}")

    # Return metrics
    metrics = {
        'accuracy': report['accuracy'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score'],
        'precision': report['weighted avg']['precision']
    }

    return metrics
