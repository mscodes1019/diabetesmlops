import argparse
import os
import glob
import pandas as pd
import mlflow

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib


def main(args):
    # Enable MLflow autologging
    mlflow.autolog()

    # Read data
    df = get_csvs_df(args.training_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Train model
    model = train_model(args.reg_rate, X_train, y_train)

    # Evaluate (basic example)
    score = model.score(X_test, y_test)
    print(f"Test accuracy: {score:.4f}")

    # Save model to outputs (captured by Azure ML)
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/model.joblib")
    print("Model saved to outputs/model.joblib")


def get_csvs_df(path):
    """Load data from either a single CSV file or all CSVs in a folder."""
    if not os.path.exists(path):
        raise RuntimeError(
            f"Cannot use non-existent path provided: {path}"
        )

    p = Path(path)
    if p.is_file():
        return pd.read_csv(p)
    else:
        csv_files = glob.glob(f"{path}/*.csv")
        if not csv_files:
            raise RuntimeError(
                f"No CSV files found in provided data path: {path}"
            )
        return pd.concat(
            (pd.read_csv(f) for f in csv_files), ignore_index=True
        )


def split_data(df):
    """Split dataframe into train/test sets."""
    # Replace "Diabetic" with your actual target column name
    X = df.drop(
        columns=[
            "Diabetic"
        ]
    )
    y = df["Diabetic"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(reg_rate, X_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression(C=1/reg_rate, solver="liblinear")
    model.fit(X_train, y_train)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_data",
        type=str,
        required=True,
        help=(
            "Path to training data "
            "(CSV file or folder containing CSVs)"
        ),
    )
    parser.add_argument(
        "--reg_rate",
        type=float,
        default=0.01,
        help=(
            "Regularization rate "
            "(inverse of C in LogisticRegression)"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("\n" + "*" * 60)
    args = parse_args()
    main(args)
    print("*" * 60 + "\n")
