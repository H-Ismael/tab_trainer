from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import logging
from typing import Optional
from training import train_svc, train_xgboost
import subprocess

# Todo: - where file is uploaded - where file is saved - where file is deleted
# Todo: - stop point of training (treshold on metric and or training number)
# Todo: - metrics to optimize

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU
def check_gpu_available():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(result.stdout.decode('utf-8'))
        if result.returncode == 0:
            logger.info("GPU is available.")
            return True
        else:
            logger.info("GPU not available.")
            return False
    except FileNotFoundError:
        logger.info("nvidia-smi command not found. GPU not available.")
        return False

gpu_available = check_gpu_available()

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
        # check_gpu_available()
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
            hyperparams,
            gpu_available
        )
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.exception("An error occurred during XGBoost training.")
        return JSONResponse(content={"error": str(e)}, status_code=500)
