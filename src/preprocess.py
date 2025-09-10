import argparse
from pathlib import Path
from .utils import read_heart_csv, basic_clean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to heart.csv (raw Kaggle file)")
    parser.add_argument("--out", required=True, help="Path to save processed.csv")
    args = parser.parse_args()

    df = read_heart_csv(args.input)
    df = basic_clean(df)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[preprocess] Saved cleaned dataset to {out_path} with shape {df.shape}")

if __name__ == "__main__":
    main()
