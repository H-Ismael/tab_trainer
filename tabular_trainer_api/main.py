from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import logging
from typing import Optional
from training import train_svc, train_xgboost
import torch

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

@app.post("/train/svc")
async def train_svc_endpoint(
    data_file: UploadFile = File(...),
    label_column: str = Form(...),
    test_column: Optional[str] = Form(None),
    test_size: float = Form(0.2),
    random_state: int = Form(42),
    use_optuna: bool = Form(False),
    hyperparams: Optional[str] = Form(None)  # JSON string of hyperparameters
):
    try:
        # Call training function
        metrics = train_svc(
            data_file.file,
            label_column,
            test_column,
            test_size,
            random_state,
            use_optuna,
            hyperparams
        )
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.exception("An error occurred during SVC training.")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/train/xgboost")
async def train_xgboost_endpoint(
    data_file: UploadFile = File(...),
    label_column: str = Form(...),
    test_column: Optional[str] = Form(None),
    test_size: float = Form(0.2),
    random_state: int = Form(42),
    use_optuna: bool = Form(False),
    hyperparams: Optional[str] = Form(None)  # JSON string of hyperparameters
):
    try:
        # Call training function
        metrics = train_xgboost(
            data_file.file,
            label_column,
            test_column,
            test_size,
            random_state,
            use_optuna,
            hyperparams
        )
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.exception("An error occurred during XGBoost training.")
        return JSONResponse(content={"error": str(e)}, status_code=500)
