import os
import pytest

@pytest.fixture(scope="session", autouse=True)
def model_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("credit")
    os.environ["CREDIT_MODEL_DIR"] = str(root / "models")
    os.environ["CREDIT_LOG_PATH"] = str(root / "decisions.csv")
    from src.models import train
    train.MODEL_DIR = root / "models"
    train.MODEL_DIR.mkdir(exist_ok=True)
    from src.monitoring import metrics
    from src.tracking import logger
    metrics.LOG_PATH = root / "decisions.csv"
    logger.LOG_PATH = metrics.LOG_PATH
    train.train()
