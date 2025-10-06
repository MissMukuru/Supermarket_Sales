from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import typer

from Supermarket_sales.config import FIGURES_DIR, PROCESSED_DATA_DIR

FIGURES_DIR = Path("C:\Users\USER\Desktop\Supermarket_Sales\notebooks\reports\figures")

app = typer.Typer()


@app.command()
def main(
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    output_dir: Path = FIGURES_DIR,
):
    """
    Generate plots to visualize model evaluation.
    """
    logger.info(f"Loading predictions from {predictions_path}...")
    df = pd.read_csv(predictions_path)

    logger.info(f"Columns in dataset: {list(df.columns)}")

    if "Target_Total" in df.columns and "Predicted_Total" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df["Target_Total"], y=df["Predicted_Total"], alpha=0.6)
        plt.plot([df["Target_Total"].min(), df["Target_Total"].max()],
                 [df["Target_Total"].min(), df["Target_Total"].max()],
                 color="red", linestyle="--")
        plt.title("Predicted vs Actual Total (Regression)")
        plt.xlabel("Actual Total")
        plt.ylabel("Predicted Total")
        plt.grid(True)
        path_reg = output_dir / "predicted_vs_actual.png"
        plt.savefig(path_reg)
        plt.close()
        logger.success(f"Saved regression plot: {path_reg}")

    if "HighSpender" in df.columns and "Predicted_HighSpender" in df.columns:
        plt.figure(figsize=(6, 5))
        cm = pd.crosstab(df["HighSpender"], df["Predicted_HighSpender"])
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        plt.title("Confusion Matrix - High Spender Classification")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        path_clf = output_dir / "confusion_matrix.png"
        plt.savefig(path_clf)
        plt.close()
        logger.success(f"Saved classification plot: {path_clf}")

    logger.success("All plots generated successfully!")


if __name__ == "__main__":
    app()
