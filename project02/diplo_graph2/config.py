from pathlib import Path

RANDOM_SEED = 42

# paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"

SAMPLE_DATA_DIR = RAW_DATA_DIR / "sample"

CASES_CSV = SAMPLE_DATA_DIR / "cases.csv"
DATA_DICT_CSV = SAMPLE_DATA_DIR / "data_dictionary.csv"
METADATA_CSV = SAMPLE_DATA_DIR / "metadata.csv"
