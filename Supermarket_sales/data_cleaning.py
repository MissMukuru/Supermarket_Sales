from pathlib import Path
import pandas as pd
from loguru import logger
from Supermarket_sales.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def clean_data(
    input_path: Path = RAW_DATA_DIR / "supermarkt_sales.xlsx",
    output_path: Path = PROCESSED_DATA_DIR / "Sales.csv"
) -> None:
    """
    Reads the raw Excel dataset, renames columns for consistency,
    and saves a clean CSV file.
    """
    logger.info("Reading raw Excel data...")
    data = pd.read_excel(input_path)

    data = data.rename(columns={
        "Invoice ID": "Invoice_ID",
        "Product line": "Product_Line",
        "Unit price": "Unit_Price",
        "Tax 5%": "Tax_5%",
        "gross margin percentage ": "Gross_Margin_Percentage",
        "gross income": "Gross_Income",
    })


    data.to_csv(output_path, index=False)
    logger.success(f"Cleaned dataset saved to {output_path}")


if __name__ == '__main__':
    clean_data()