import warnings
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor


def prepare_dataset(csv_path: str = "dataset.csv") -> tuple[pd.DataFrame, pd.Series]:
    """Load, clean, and engineer features required for Milestone 3 models."""
    df = pd.read_csv(csv_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["region"] = (
        df["region"].str.strip().str.lower().str.replace(" ", "-", regex=False)
    )
    df["region"] = df["region"].replace(
        {
            "us-east": "US-East",
            "us-west": "US-West",
            "india-south": "India-South",
        }
    )

    df = df.drop_duplicates()

    df["usage_units"] = df["usage_units"].interpolate()
    df["provisioned_capacity"] = df["provisioned_capacity"].interpolate()
    df["cost_usd"] = df["cost_usd"].fillna(df["usage_units"] * 0.1)
    df["availability_pct"] = df["availability_pct"].ffill()
    df["is_holiday"] = df["is_holiday"].fillna(0)
    df["economic_index"] = df["economic_index"].interpolate()
    df["market_demand_index"] = df["market_demand_index"].fillna(
        df["market_demand_index"].rolling(7, min_periods=1).mean()
    )
    df["temperature_index"] = df["temperature_index"].interpolate()

    # Feature engineering from Milestone 2 and 3 notebook steps.
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["lag_1_usage"] = df["usage_units"].shift(1)
    df["lag_7_usage"] = df["usage_units"].shift(7)
    df["rolling_mean_3"] = df["usage_units"].rolling(window=3).mean()

    threshold = df["usage_units"].mean() + df["usage_units"].std()
    df["usage_spike"] = np.where(df["usage_units"] > threshold, 1, 0)

    df = pd.get_dummies(df, columns=["region", "service_type"], drop_first=True)

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day_of_month"] = df["timestamp"].dt.day

    df = df.drop(columns=["timestamp"])

    # Backfill handles initial lag/rolling NaNs so all models can train cleanly.
    df = df.bfill().ffill()

    X = df.drop("usage_units", axis=1)
    y = df["usage_units"]
    return X, y


def train_test_split_time(
    X: pd.DataFrame, y: pd.Series, ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_size = int(len(X) * ratio)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    return X_train, X_test, y_train, y_test


def arima_grid_search(
    y_train: pd.Series, y_test: pd.Series
) -> tuple[ARIMA, np.ndarray, float, tuple[int, int, int]]:
    p = range(0, 4)
    d = range(0, 2)
    q = range(0, 4)

    best_score = float("inf")
    best_order = None

    for i in p:
        for j in d:
            for k in q:
                try:
                    model = ARIMA(y_train, order=(i, j, k))
                    model_fit = model.fit()
                    pred = model_fit.forecast(steps=len(y_test))
                    rmse = np.sqrt(mean_squared_error(y_test, pred))

                    if rmse < best_score:
                        best_score = rmse
                        best_order = (i, j, k)
                except Exception:
                    continue

    if best_order is None:
        best_order = (1, 1, 1)

    best_model = ARIMA(y_train, order=best_order).fit()
    best_pred = best_model.forecast(steps=len(y_test))
    best_rmse = np.sqrt(mean_squared_error(y_test, best_pred))

    return best_model, np.asarray(best_pred), best_rmse, best_order


def xgboost_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[XGBRegressor, np.ndarray, float, dict]:
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0],
    }

    grid_search = GridSearchCV(
        estimator=XGBRegressor(objective="reg:squarederror", random_state=42),
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=1,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_xgb = grid_search.best_estimator_
    pred = best_xgb.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    return best_xgb, pred, rmse, grid_search.best_params_


def main() -> None:
    warnings.filterwarnings("ignore")

    X, y = prepare_dataset("dataset.csv")
    X_train, X_test, y_train, y_test = train_test_split_time(X, y)

    # Baseline models
    arima_model = ARIMA(y_train, order=(1, 1, 1)).fit()
    arima_pred = arima_model.forecast(steps=len(y_test))

    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        objective="reg:squarederror",
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    rmse_arima = np.sqrt(mean_squared_error(y_test, arima_pred))
    rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb_pred))

    print("Baseline ARIMA RMSE:", rmse_arima)
    print("Baseline XGBoost RMSE:", rmse_xgb)

    # Tuned models
    best_arima_model, arima_tuned_pred, rmse_arima_tuned, best_order = arima_grid_search(
        y_train, y_test
    )
    best_xgb_model, xgb_tuned_pred, rmse_xgb_tuned, best_params = xgboost_grid_search(
        X_train, y_train, X_test, y_test
    )

    print("\nBest ARIMA order:", best_order)
    print("Best XGBoost params:", best_params)
    print("Tuned ARIMA RMSE:", rmse_arima_tuned)
    print("Tuned XGBoost RMSE:", rmse_xgb_tuned)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    with open(artifacts_dir / "tuned_xgboost_model.pkl", "wb") as xgb_file:
        pickle.dump(
            {
                "model": best_xgb_model,
                "feature_columns": X_train.columns.tolist(),
                "best_params": best_params,
            },
            xgb_file,
        )

    with open(artifacts_dir / "tuned_arima_model.pkl", "wb") as arima_file:
        pickle.dump(
            {
                "model": best_arima_model,
                "order": best_order,
            },
            arima_file,
        )

    print(f"Saved XGBoost PKL: {artifacts_dir / 'tuned_xgboost_model.pkl'}")
    print(f"Saved ARIMA PKL: {artifacts_dir / 'tuned_arima_model.pkl'}")

    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual")
    plt.plot(arima_tuned_pred, label="ARIMA")
    plt.plot(xgb_tuned_pred, label="XGBoost")
    plt.legend()
    plt.title("Demand Forecast Comparison")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()