import joblib
import sys
import pandas as pd
from yaml import safe_load
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path


TARGET = "trip_duration"


def load_dataframe(path):
    df = pd.read_csv(path)
    return df


def make_X_y(dataframe: pd.DataFrame, target_column: str):
    df_copy = dataframe.copy()

    X = df_copy.drop(columns=[target_column])
    y = df_copy[target_column]

    return X, y


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def save_model(model, save_path):
    joblib.dump(value=model, filename=save_path)


def load_params(params_path: Path, model_name: str):
    with open(params_path, "r") as f:
        params = safe_load(f)

    model_params = params["train_model"][model_name]
    return model_params


def get_model(model_name: str, model_params: dict):
    if model_name == "xgboost":
        return XGBRegressor(**model_params)

    elif model_name == "random_forest_regressor":
        return RandomForestRegressor(**model_params)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def main():
    # current file path
    current_path = Path(__file__).resolve()

    # root directory path
    root_path = current_path.parent.parent.parent

    # read input file path from command line
    training_data_path = root_path / sys.argv[1]

    # load params.yaml from project root
    params_path = root_path / "params.yaml"

    # choose which model config to use
    # model_name = "xgboost"
    model_name = "random_forest_regressor"

    # load the data
    train_data = load_dataframe(training_data_path)

    # split the data into X and y
    X_train, y_train = make_X_y(dataframe=train_data, target_column=TARGET)

    # load hyperparameters from yaml
    model_params = load_params(params_path=params_path, model_name=model_name)

    # create model with yaml hyperparameters
    regressor = get_model(model_name=model_name, model_params=model_params)

    # train model
    regressor = train_model(
        model=regressor,
        X_train=X_train,
        y_train=y_train,
    )

    # save trained model
    model_output_path = root_path / "models" / "models"
    model_output_path.mkdir(exist_ok=True)

    save_model(
        model=regressor,
        save_path=model_output_path / f"{model_name}.joblib",
    )


if __name__ == "__main__":
    main()