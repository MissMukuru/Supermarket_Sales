# src/features/build_features.py

from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer
import os

from Supermarket_sales.config import PROCESSED_DATA_DIR

app = typer.Typer()


def simple_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df_encoded


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "Clean_data.csv",
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
):
    """
    Generate ML-ready features and labels from cleaned dataset.
    """
    # Ensure output directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Missing input file at {input_path}")

    logger.info(f"Loading cleaned dataset from {input_path}")
    df = pd.read_csv(input_path)

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Time"] = df["Time"].astype(str)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.day_name()
    df["IsWeekend"] = df["Weekday"].isin(["Saturday", "Sunday"]).astype(int)
    df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour
    df["PartOfTheDay"] = pd.cut(
        df["Hour"],
        bins=[-1, 11, 16, 20, 24],
        labels=["Morning", "Afternoon", "Evening", "Night"],
    )

    df["Target_Total"] = df["Total"]  # regression
    df["HighSpender"] = (df["Total"] > 500).astype(int)  # classification
    df["Average_price_Item"] = df["Total"] / df["Quantity"]

    cols_to_drop = [
        "Invoice ID",
        "gross margin percentage ",
        "Tax 5%",
        "cogs",
        "Date",
        "Time",
    ]
    df_model = df.drop(columns=cols_to_drop, errors="ignore")

    # --- Split into features and labels ---
    labels = df_model[["Target_Total", "HighSpender"]]
    features = df_model.drop(columns=["Target_Total", "HighSpender"])

    # --- Encode categorical variables ---
    features_encoded = simple_encode(features)

    # --- Save CSVs ---
    logger.info("Saving processed features and labels...")
    for _ in tqdm(range(1), desc="Saving CSVs"):
        features_encoded.to_csv(features_path, index=False)
        labels.to_csv(labels_path, index=False)

    logger.success(f"Features saved to {features_path}")
    logger.success(f"Labels saved to {labels_path}")


if __name__ == "__main__":
    app()
