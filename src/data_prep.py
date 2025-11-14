import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--raw_data", type=str, required=True)
parser.add_argument("--prep_data_dir", type=str, required=True)
args = parser.parse_args()

iris_df = pd.read_csv(args.raw_data)

if "Id" in iris_df.columns:
    iris_df = iris_df.drop("Id", axis=1)

species_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
iris_df["Species"] = iris_df["Species"].map(species_map)

train_df, test_df = train_test_split(
    iris_df, test_size=0.2, random_state=42, stratify=iris_df["Species"]
)

Path(args.prep_data_dir).mkdir(parents=True, exist_ok=True)
train_df.to_csv(Path(args.prep_data_dir) / "train.csv", index=False)
test_df.to_csv(Path(args.prep_data_dir) / "test.csv", index=False)
