import argparse
import mlflow
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
args = parser.parse_args()

model_path = args.model_dir

print(f"Registering model from: {model_path}")
mlflow.register_model(
    model_uri=f"file://{model_path}",
    name=args.model_name
)

print(f"Successfully registered model '{args.model_name}'")