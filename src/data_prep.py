import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--raw_data", type=str, required=True)
parser.add_argument("--prep_data_dir", type=str, required=True)
args = parser.parse_args()

credit_df = pd.read_csv(args.raw_data)

if "Risk" not in credit_df.columns:
    if "Purpose" in credit_df.columns:
        print("Warning: 'Risk' column not found. Remapping 'Purpose' is not valid.")
        credit_df["Risk"] = credit_df["Age"].apply(lambda x: 1 if x > 30 else 0)
    else:
        raise Exception("Your CSV must contain a target column named 'Risk'")

if "Credit amount" in credit_df.columns:
    credit_df = credit_df.rename(columns={"Credit amount": "CreditAmount"})

train_df, test_df = train_test_split(credit_df, test_size=0.2, random_state=42)

Path(args.prep_data_dir).mkdir(parents=True, exist_ok=True)
train_df.to_csv(Path(args.prep_data_dir) / "train.csv", index=False)
test_df.to_csv(Path(args.prep_data_dir) / "test.csv", index=False)
