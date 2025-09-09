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