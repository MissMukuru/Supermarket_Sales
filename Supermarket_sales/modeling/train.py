from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import joblib
from pathlib import path
from loguru import logger
from tqdm import tqdm
import typer

from Supermarket_sales.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
     features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    regression_model_path: Path = MODELS_DIR / "Random_forest_regression_model.pkl",
    classification_model_path: Path = MODELS_DIR / "Random_forest_classifier_model.pkl"

 ):
     logger.info("Loading features and labels.....")
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)

    y_reg =y['Target_Total']
    y_clf =y['HighSpender']

    logger.info("Splitting the data into train/Test sets")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    #Regression
    logger.info("Training the regression model...")
    reg_model=RandomForestRegressor(n_estimators= 100,random_state=42)
    reg_model.fit(X_train_r, y_train_r)
    y_pred_r=reg_model.predict(X_test_r)

    logger.info(f"Regression R2: {r2_score(y_test_r, y_pred_r):.4f}")
    logger.info(f"Regression RMSE: {mean_squared_error(y_test_r, y_pred_r, squared=False):.4f}")

    joblib.dump(reg_model,regression_model_path)
    logger.success(f'Regression Model saved to {regression_model_path}')

    #Classification
    logger.info("Training Random Forest Classifier....")
    clf_model= RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train_c, y_train_c)
    y_pred_c=clf_model.predict(X_test_c)

    logger.info(f"Classification Accuracy: {accuracy_score(y_test_c, y_pred_c):.4f}")
    logger.info(f"\n{classification_report(y_test_c, y_pred_c)}")
    
    joblib.dump(clf_model, classification_model_path)
    logger.success(f"Classification model saved to {classification_model_path}")

    logger.success("All modeling complete.")
if __name__ == "__main__":
    app()
