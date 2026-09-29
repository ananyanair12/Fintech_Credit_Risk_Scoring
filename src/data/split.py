"""Load the Kaggle training file, run a data-quality report, and create stratified splits."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import COLUMN_MAP, PROCESSED, RAW, SEED, TARGET


def load_raw() -> pd.DataFrame:
    """Read cs-training.csv and rename columns to snake_case."""
    path = RAW / "cs-training.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put the Kaggle 'Give Me Some Credit' files in data/raw/."
        )
    df = pd.read_csv(path, index_col=0)
    return df.rename(columns=COLUMN_MAP)


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


def make_splits() -> None:
    df = load_raw()
    print(f"Shape: {df.shape}")
    print(f"Default rate: {df[TARGET].mean():.4f}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\nMissing values:")
    print(df.isna().sum().to_string())

    train, temp = train_test_split(
        df, test_size=0.30, stratify=df[TARGET], random_state=SEED
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp[TARGET], random_state=SEED
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    print()
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_parquet(PROCESSED / f"{name}.parquet")
        print(f"{name:>5}: {len(part):>7,} rows | default rate {part[TARGET].mean():.4f}")


if __name__ == "__main__":
    make_splits()