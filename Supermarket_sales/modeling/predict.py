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
    """
    Load trained models, generate predictions, and save results for evaluation & visualization.
    """
    logger.info(f"📂 Loading features from {features_path}")
    df = pd.read_csv(features_path)
    
    target_cols = [col for col in ['Target_Total', 'HighSpender'] if col in df.columns]
    X = df.drop(columns=target_cols, errors="ignore")

    logger.info("🧠 Loading models...")
    reg_model = joblib.load(regression_model_path)
    clf_model = joblib.load(classification_model_path)

    logger.info("🔮 Generating predictions...")
    y_reg_pred = reg_model.predict(X)
    y_clf_pred = clf_model.predict(X)

    logger.info(f"💾 Saving predictions to {predictions_path}")
    df_predictions = df.copy()

    # Include true targets if they exist
    if 'Target_Total' in df.columns:
        df_predictions['Target_Total'] = df['Target_Total']
    if 'HighSpender' in df.columns:
        df_predictions['HighSpender'] = df['HighSpender']

    # Add predictions
    df_predictions['Predicted_Total'] = y_reg_pred
    df_predictions['Predicted_HighSpender'] = y_clf_pred

    df_predictions.to_csv(predictions_path, index=False)

    logger.success(f"Prediction complete! File saved at: {predictions_path}")


if __name__ == "__main__":
    app()
