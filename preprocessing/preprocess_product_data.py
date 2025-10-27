import pandas as pd
from pathlib import Path

def preprocess_fashion_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"The file at {csv_path} does not exist.")

    # Load
    df = pd.read_csv(path)
    original = len(df) # Testing

    # Clean
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    cleaned = len(df) # Testing

    # Save (For testing)
    # output_path = path.with_name(path.stem + "_cleaned.csv")
    # df.to_csv(output_path, index=False)
    # print(f"Cleaned data saved to: {output_path}")

    # Print (For testing)
    print(f"Removed {original - cleaned} rows during preprocessing.")

    return df

if __name__ == "__main__":
    data_path = "../data/products.csv"
    processed_df = preprocess_fashion_data(data_path)
