from pathlib import Path

import joblib
import pandas as pd
import typer
from loguru import logger

from Supermarket_sales.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    regression_model_path: Path = MODELS_DIR / "Random_forest_regression_model.pkl",
    classification_model_path: Path = MODELS_DIR / "Random_forest_classifier_model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    logger.info(f'loading features from {features_path}')
    df = pd.read_csv(features_path)

    X = df.drop(columns=['Target_Total', 'HighSpender'], errors='ignore')

    logger.info('Loading the models')
    reg_model = joblib.load(regression_model_path)
    clf_model = joblib.load(classification_model_path)

    logger.info("Making predictions...")
    y_reg_pred = reg_model.predict(X)
    y_clf_pred = clf_model.predict(X)

    logger.info(f'Saving predictions to {predictions_path}')
    df_predictions = df.copy()
    df_predictions['Predicted_Total'] = y_reg_pred
    df_predictions['Predicted_HighSpender'] = y_clf_pred
    df_predictions.to_csv(predictions_path, index=False)

    logger.success('Prediction complete...')

if __name__ == "__main__":
    app()
