import argparse
import pandas as pd
import joblib
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--prep_data_dir", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)
args = parser.parse_args()

mlflow.autolog(log_models=False)

train_df = pd.read_csv(Path(args.prep_data_dir) / "train.csv")
test_df = pd.read_csv(Path(args.prep_data_dir) / "test.csv")

features = ["Duration", "CreditAmount", "Age"]
target = "Risk"

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
mlflow.log_metric("accuracy", accuracy)

Path(args.model_dir).mkdir(parents=True, exist_ok=True)
joblib.dump(model, Path(args.model_dir) / "model.pkl")
