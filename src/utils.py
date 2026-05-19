EXCLUDE_OUTLIER_COLS = [
    "latitude",
    "longitude",
    "year",
    "month",
    "day",
    "hour",
    "day_of_week",
]

SKEWED_COLS = [
    "precip_mm",
    "gust_kph",
    "wind_kph",
    "air_quality_PM2.5",
    "air_quality_PM10",
]


def check_min_value(df, column, min_value):
    return (df[column] < min_value).sum()


def check_max_value(df, column, max_value):
    return (df[column] > max_value).sum()


def check_range(df, column, min_value, max_value):
    return (
        ((df[column] < min_value) |
         (df[column] > max_value))
    ).sum()

def get_categorical_columns(df):
    return df.select_dtypes(include=["object"]).columns.tolist()