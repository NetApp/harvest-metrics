import argparse
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constants
MODEL_PATH = "models/gbr_model.pkl"
SCALER_PATH = "models/gbr_scaler.pkl"

BYTES_PER_MD = 1024 * 1024
POLLER = "Poller"
PLUS_SOME = 1.2  # 20% more memory than predicted
PREDICTED_MB = "EstimatedMB"
PREDICTED_RSS = "PredictedRss"
RSS_BYTES = "RssBytes"
RSS_MB = "RssMB"
HARVESTED_FEATURES = [
    "DiskConfig",
    "DiskPerf",
    "LunConfig",
    "LunPerf",
    "NFSClientsConfig",
    "QtreeConfig",
    "QtreePerf",
    "SVMConfig",
    "SensorConfig",
    "SnapMirrorConfig",
    "SnapshotConfig",
    "StorageGridSG",
    "VolumeAnalyticsConfig",
    "VolumeConfig",
    "VolumePerf",
    "WorkloadDetailVolumePerf",
]


def train_model(args):
    # check that the input file exists
    if not args.input.exists():
        print(f'Error: The input file "{args.input}" does not exist.')
        return

    # Load the CSV file
    try:
        data = pd.read_csv(args.input)
    except Exception as e:
        print(f'Error: Failed to read the input file "{args.input}". {e}')
        return

    # Prepare the data using the selected features
    x_selected = data[HARVESTED_FEATURES]
    y = data[RSS_BYTES]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_selected, y, test_size=0.2, random_state=42
    )

    # Standardize the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    x_selected_scaled = scaler.transform(x_selected)

    # Set common parameters for GradientBoostingRegressor
    params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "subsample": 0.9,
        "max_features": 0.9,
        "random_state": 42,
    }

    # Train the model with the common parameters
    gbr = GradientBoostingRegressor(**params)
    gbr.fit(x_train_scaled, y_train)

    # Save the model and scaler to disk
    model_file_path = MODEL_PATH
    scaler_file_path = SCALER_PATH
    joblib.dump(gbr, model_file_path)
    joblib.dump(scaler, scaler_file_path)

    # Predict using the loaded GradientBoostingRegressor model
    data[PREDICTED_RSS] = gbr.predict(x_selected_scaled)
    y_train_pred = gbr.predict(x_train_scaled)
    y_test_pred = gbr.predict(x_test_scaled)

    # Create new columns RssMB and PredictedMB
    data[RSS_MB] = data[RSS_BYTES] / BYTES_PER_MD
    data[PREDICTED_MB] = data[PREDICTED_RSS] / BYTES_PER_MD

    # Save the updated DataFrame to a new CSV file
    if args.save:
        output_file_path = args.save
        data.to_csv(output_file_path, index=False)

    # Evaluate the model performance on training data
    r2_train = r2_score(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

    # Evaluate the model performance on test data
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Display the first few rows of the updated DataFrame to verify the results
    print(data[[RSS_BYTES, RSS_MB, PREDICTED_RSS, PREDICTED_MB]].head())

    # Create a DataFrame to store evaluation metrics
    metrics_df = pd.DataFrame(
        {
            "Dataset": ["Training", "Test"],
            "R^2": [r2_train, r2_test],
            "MAE": [mae_train, mae_test],
            "RMSE": [rmse_train, rmse_test],
        }
    )

    print("\nModel evaluation metrics:")
    # Print the DataFrame
    print(metrics_df)


# Validate input
def validate_input(df):
    if POLLER not in df.columns:
        print('Error: The DataFrame does not contain a "Poller" column.')
        return False

    is_valid = True
    nan_indices = np.where(pd.isna(df))

    if len(nan_indices[0]) == 0:
        return True

    for row, col in zip(*nan_indices):
        poller_value = df.loc[row, POLLER]
        poller_name = "unnamed" if pd.isna(poller_value) else poller_value
        print(f'Poller "{poller_name}" is missing the required key: {df.columns[col]}')
        is_valid = False

    return is_valid


def predict_size(args):
    # check that the input file exists
    if not args.input.exists():
        print(f'Error: The input file "{args.input}" does not exist.')
        return

    # Load the input JSON file
    try:
        input_data = pd.read_json(args.input)
    except Exception as e:
        print(f'Error: Failed to read the input file "{args.input}". {e}')
        return

    is_valid = validate_input(input_data)
    if not is_valid:
        return

    # Load the model and scaler
    model_file_path = MODEL_PATH
    scaler_file_path = SCALER_PATH
    gbr = joblib.load(model_file_path)
    scaler = joblib.load(scaler_file_path)

    # Prepare the input data using the selected features
    x_input = input_data[HARVESTED_FEATURES]
    x_input_scaled = scaler.transform(x_input)

    # Predict the memory size and add 20% (PLUS_SOME) more memory
    input_data[PREDICTED_RSS] = gbr.predict(x_input_scaled)
    input_data[PREDICTED_MB] = input_data[PREDICTED_RSS] * PLUS_SOME / BYTES_PER_MD

    # Round the predicted memory size to the nearest integer and print with no decimals
    input_data[PREDICTED_MB] = input_data[PREDICTED_MB].round(0).astype(int)

    # Calculate the total predicted memory size
    total_predicted_mb = input_data[PREDICTED_MB].sum()

    # Add a summary row to the DataFrame
    summary_row = pd.DataFrame({
        POLLER: "Total",
        PREDICTED_MB: [total_predicted_mb],
    })
    input_data = pd.concat([input_data, summary_row], ignore_index=True)

    # Left justify the poller column
    input_data[POLLER] = input_data[POLLER].apply(lambda x: f"{x:<}")

    # Display the input data with the predicted memory size
    print(input_data[[POLLER, PREDICTED_MB]].to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Planner")
    sub_parsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create a parser for the train command
    train_parser = sub_parsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "-i",
        "--input",
        type=pathlib.Path,
        required=True,
        help="CSV file with the training data",
    )
    train_parser.add_argument(
        "-s",
        "--save",
        type=pathlib.Path,
        help="Path of to save the input file with predictions",
    )

    # Create a parser for the estimateMemory command
    predict_parser = sub_parsers.add_parser(
        "estimate-memory", help="Estimate the amount of memory needed"
    )
    predict_parser.add_argument(
        "-i",
        "--input",
        type=pathlib.Path,
        required=True,
        help="Object counts JSON file from bin/harvest planner",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_model(args)
    elif args.command == "estimate-memory":
        predict_size(args)
    else:
        parser.print_help()


def main():
    parse_args()


if __name__ == "__main__":
    main()
