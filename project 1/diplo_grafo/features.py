from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from diplo_grafo.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    """
    Extract features from the dataset and save them to a CSV file.

    Args:
        input_path (Path): Path to the input dataset CSV file.
        output_path (Path): Path to save the extracted features CSV file.
    """
    logger.info(f"Loading dataset from {input_path}")
    # Load your dataset here (e.g., using pandas)
    # df = pd.read_csv(input_path)

    logger.info("Extracting features...")
    # Perform feature extraction here
    # features_df = extract_features(df)

    logger.info(f"Saving extracted features to {output_path}")
    # Save the features DataFrame to CSV
    # features_df.to_csv(output_path, index=False)



    


if __name__ == "__main__":
    app()
