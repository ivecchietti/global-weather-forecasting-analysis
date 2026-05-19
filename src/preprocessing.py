import numpy as np
import pandas as pd

from src import utils as ut


def sort_temporal_data(
    df,
    group_col="location_name",
    date_col="last_updated",
):
    return (
        df
        .sort_values([group_col, date_col])
        .reset_index(drop=True)
        .copy()
    )


def add_cyclical_features(df):
    df = df.copy()

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )
    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df


def add_lag_features(
    df,
    group_col="location_name",
    target_col="temperature_celsius",
):
    df = df.copy()

    grouped_target = df.groupby(group_col)[target_col]

    df["temp_lag_1"] = grouped_target.shift(1)
    df["temp_lag_3"] = grouped_target.shift(3)
    df["temp_lag_24"] = grouped_target.shift(24)

    df["temp_roll_3"] = grouped_target.transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    return df


def run_physical_checks(df, checks):
    results = []

    for check in checks:
        column = check["column"]
        check_type = check["type"]

        if column not in df.columns:
            raise ValueError(f"Column not found: {column}")

        if check_type == "min":
            count = ut.check_min_value(
                df,
                column,
                check["value"],
            )

            check_name = f"{column} < {check['value']}"

        elif check_type == "max":
            count = ut.check_max_value(
                df,
                column,
                check["value"],
            )

            check_name = f"{column} > {check['value']}"

        elif check_type == "range":
            count = ut.check_range(
                df,
                column,
                check["min"],
                check["max"],
            )

            check_name = (
                f"{column} outside "
                f"[{check['min']}, {check['max']}]"
            )

        else:
            raise ValueError(f"Unknown check type: {check_type}")

        results.append({
            "check": check_name,
            "count": count,
        })

    return (
        pd.DataFrame(results)
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def temporal_train_val_test_split(
    df,
    dev_ratio=0.8,
    train_ratio=0.8,
    date_col="last_updated",
):
    df = (
        df
        .sort_values(date_col)
        .reset_index(drop=True)
        .copy()
    )

    dev_size = int(len(df) * dev_ratio)

    dev_df = df.iloc[:dev_size].copy()
    test_df = df.iloc[dev_size:].copy()

    train_size = int(len(dev_df) * train_ratio)

    train_df = dev_df.iloc[:train_size].copy()
    val_df = dev_df.iloc[train_size:].copy()

    return train_df, val_df, test_df


def detect_outliers_iqr(train_df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ut.EXCLUDE_OUTLIER_COLS

    numeric_cols = [
        col
        for col in train_df.select_dtypes(
            include=["int64", "float64", "int32"]
        ).columns
        if col not in exclude_cols
    ]

    outlier_summary = []

    for col in numeric_cols:
        q1 = train_df[col].quantile(0.25)
        q3 = train_df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        n_outliers = (
            (train_df[col] < lower) |
            (train_df[col] > upper)
        ).sum()

        outlier_summary.append({
            "feature": col,
            "lower_bound": lower,
            "upper_bound": upper,
            "outliers": n_outliers,
            "percentage": n_outliers / len(train_df) * 100,
        })

    return (
        pd.DataFrame(outlier_summary)
        .sort_values("percentage", ascending=False)
        .reset_index(drop=True)
    )


def add_log_features(train_df, val_df, test_df, skewed_cols=None):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    if skewed_cols is None:
        skewed_cols = ut.SKEWED_COLS

    safe_log_cols = []

    for col in skewed_cols:
        if col not in train_df.columns:
            raise ValueError(f"Column not found: {col}")

        if train_df[col].min() >= 0:
            safe_log_cols.append(col)
        else:
            print(f"Skipping {col}: contains negative values in train.")

    for col in safe_log_cols:
        train_df[f"{col}_log"] = np.log1p(train_df[col])
        val_df[f"{col}_log"] = np.log1p(
            val_df[col].clip(lower=0)
        )
        test_df[f"{col}_log"] = np.log1p(
            test_df[col].clip(lower=0)
        )

    return train_df, val_df, test_df, safe_log_cols

def convert_time_to_minutes(time_str):
    if time_str.startswith("No "):
        return np.nan

    parsed_time = pd.to_datetime(
        time_str,
        format="%I:%M %p",
    )

    return parsed_time.hour * 60 + parsed_time.minute


def add_sun_time_features(df):
    df = df.copy()

    time_cols = [
        "sunrise",
        "sunset",
    ]

    for col in time_cols:
        df[f"{col}_minutes"] = (
            df[col]
            .apply(convert_time_to_minutes)
        )

    return df


def add_wind_direction_features(df):
    df = df.copy()

    wind_mapping = {
        "N": 0,
        "NNE": 22.5,
        "NE": 45,
        "ENE": 67.5,
        "E": 90,
        "ESE": 112.5,
        "SE": 135,
        "SSE": 157.5,
        "S": 180,
        "SSW": 202.5,
        "SW": 225,
        "WSW": 247.5,
        "W": 270,
        "WNW": 292.5,
        "NW": 315,
        "NNW": 337.5,
    }

    df["wind_direction_degrees"] = (
        df["wind_direction"]
        .map(wind_mapping)
    )

    df["wind_dir_sin"] = np.sin(
        2 * np.pi * df["wind_direction_degrees"] / 360
    )

    df["wind_dir_cos"] = np.cos(
        2 * np.pi * df["wind_direction_degrees"] / 360
    )

    return df


def one_hot_encode(df, columns, drop_first=True):
    df = df.copy()

    return pd.get_dummies(
        df,
        columns=columns,
        drop_first=drop_first,
    )

def frequency_encode(
    train_df,
    val_df,
    test_df,
    columns,
):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    encoding_maps = {}

    for col in columns:

        freq_map = (
            train_df[col]
            .value_counts(normalize=True)
        )

        encoding_maps[col] = freq_map

        train_df[f"{col}_freq"] = (
            train_df[col]
            .map(freq_map)
        )

        val_df[f"{col}_freq"] = (
            val_df[col]
            .map(freq_map)
            .fillna(0)
        )

        test_df[f"{col}_freq"] = (
            test_df[col]
            .map(freq_map)
            .fillna(0)
        )

    return (
        train_df,
        val_df,
        test_df,
        encoding_maps,
    )

def scale_features(
    train_df,
    val_df,
    test_df,
    scaling_config,
):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    scaler_stats = {}

    for method, columns in scaling_config.items():

        for col in columns:

            if col not in train_df.columns:
                raise ValueError(f"Column not found: {col}")

            if method == "standard":
                mean = train_df[col].mean()
                std = train_df[col].std()

                if std == 0 or pd.isna(std):
                    print(f"Skipping {col}: std is 0 or NaN.")
                    continue

                train_df[col] = (train_df[col] - mean) / std
                val_df[col] = (val_df[col] - mean) / std
                test_df[col] = (test_df[col] - mean) / std

                scaler_stats[col] = {
                    "method": method,
                    "mean": mean,
                    "std": std,
                }

            elif method == "minmax":
                min_value = train_df[col].min()
                max_value = train_df[col].max()
                denominator = max_value - min_value

                if denominator == 0 or pd.isna(denominator):
                    print(f"Skipping {col}: min and max are equal or NaN.")
                    continue

                train_df[col] = (train_df[col] - min_value) / denominator
                val_df[col] = (val_df[col] - min_value) / denominator
                test_df[col] = (test_df[col] - min_value) / denominator

                scaler_stats[col] = {
                    "method": method,
                    "min": min_value,
                    "max": max_value,
                }

            elif method == "robust":
                median = train_df[col].median()
                q1 = train_df[col].quantile(0.25)
                q3 = train_df[col].quantile(0.75)
                iqr = q3 - q1

                if iqr == 0 or pd.isna(iqr):
                    print(f"Skipping {col}: IQR is 0 or NaN.")
                    continue

                train_df[col] = (train_df[col] - median) / iqr
                val_df[col] = (val_df[col] - median) / iqr
                test_df[col] = (test_df[col] - median) / iqr

                scaler_stats[col] = {
                    "method": method,
                    "median": median,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                }

            else:
                raise ValueError(f"Unknown scaling method: {method}")

    return train_df, val_df, test_df, scaler_stats