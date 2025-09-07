import pandas as pd


def get_missings(df, columns):
    """
    Hàm trả về dataframe chứa số lượng giá trị null của các cột được chỉ định trong danh sách columns
    Args:
        df (pd.DataFrame): DataFrame cần kiểm tra giá trị null
        columns (list): Danh sách tên các cột cần kiểm tra giá trị null
    Returns:
        pd.DataFrame: DataFrame chứa số lượng giá trị null của các cột được chỉ định
    """
    dict_nulls = {}
    for col in columns:
        dict_nulls[col] = df[col].isnull().sum()

    df_nulls = pd.DataFrame(
        data=list(dict_nulls.values()),
        index=list(dict_nulls.keys()),
        columns=["MissingNumber"],
    )
    return df_nulls


def get_missings_percentage(df, columns):
    """
    Hàm trả về dataframe chứa phần trăm giá trị null của các cột được chỉ định trong danh sách columns
    Args:
        df (pd.DataFrame): DataFrame cần kiểm tra giá trị null
        columns (list): Danh sách tên các cột cần kiểm tra giá trị null
    Returns:
        pd.DataFrame: DataFrame chứa phần trăm giá trị null của các cột được chỉ định
    """
    dict_nulls = {}
    for col in df.columns:
        percentage_null_values = str(round(df[col].isnull().sum() / len(df), 2)) + "%"
        dict_nulls[col] = percentage_null_values

    df_nulls = pd.DataFrame(
        data=list(dict_nulls.values()),
        index=list(dict_nulls.keys()),
        columns=["%Missing"],
    )
    return df_nulls


def detect_outliers(df, col):
    """
    Hàm phát hiện ngoại lai trong một cột số bằng phương pháp IQR
    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu
        col (str): Tên cột số cần kiểm tra ngoại lai
    Returns:
        pd.DataFrame: DataFrame chứa các hàng có giá trị ngoại lai trong cột được chỉ định
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    return outliers


def clean_outliers(df, col):
    """
    Hàm loại bỏ ngoại lai trong một cột số bằng phương pháp IQR
    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu
        col (str): Tên cột số cần loại bỏ ngoại lai
    Returns:
        pd.DataFrame: DataFrame đã loại bỏ các hàng có giá trị ngoại lai trong cột được chỉ định
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df_cleaned = df.loc[(df[col] >= lower) & (df[col] <= upper)]
    return df_cleaned
