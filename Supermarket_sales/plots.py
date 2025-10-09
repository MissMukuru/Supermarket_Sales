from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import typer

from Supermarket_sales.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    save_dir: Path = Path(__file__).resolve().parents[2] / "notebooks" / "reports" / "figures",
):
    """
    Generate plots to visualize model evaluation metrics.
    """
    logger.info(f"📂 Loading predictions from: {predictions_path}")
    df = pd.read_csv(predictions_path)

    # Ensure save directory exists
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"💾 Plots will be saved to: {save_dir.resolve()}")

    logger.info(f"📊 Columns available: {list(df.columns)}")

    # --- Regression Plot ---
    if {"Target_Total", "Predicted_Total"}.issubset(df.columns):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df["Target_Total"], y=df["Predicted_Total"], alpha=0.6)
        plt.plot(
            [df["Target_Total"].min(), df["Target_Total"].max()],
            [df["Target_Total"].min(), df["Target_Total"].max()],
            color="red", linestyle="--"
        )
        plt.title("Predicted vs Actual Total (Regression)")
        plt.xlabel("Actual Total")
        plt.ylabel("Predicted Total")
        plt.grid(True)

        path_reg = save_dir / "predicted_vs_actual.png"
        plt.savefig(path_reg, bbox_inches="tight")
        plt.close()
        logger.success(f"✅ Saved regression plot: {path_reg}")

        # Check file existence
        if path_reg.exists():
            logger.info(f"🟢 Confirmed saved: {path_reg.resolve()}")
        else:
            logger.error("❌ Regression plot NOT found after saving!")

    # --- Classification Plot ---
    if {"HighSpender", "Predicted_HighSpender"}.issubset(df.columns):
        plt.figure(figsize=(6, 5))
        cm = pd.crosstab(df["HighSpender"], df["Predicted_HighSpender"])
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        plt.title("Confusion Matrix - High Spender Classification")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        path_clf = save_dir / "confusion_matrix.png"
        plt.savefig(path_clf, bbox_inches="tight")
        plt.close()
        logger.success(f"✅ Saved classification plot: {path_clf}")

        if path_clf.exists():
            logger.info(f"🟢 Confirmed saved: {path_clf.resolve()}")
        else:
            logger.error("❌ Classification plot NOT found after saving!")

    logger.success("🎯 All plots generated successfully!")


if __name__ == "__main__":
    app()
