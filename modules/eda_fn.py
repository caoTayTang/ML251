import os
import pandas as pd
from PIL import Image


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


def build_image_dataframe(dataset_path, max_images_per_class=None):
    """
    Hàm trả về dataframe về thông tin các ảnh trong đường dẫn dataset_path
    Args:
        dataset_path(str): đường dẫn ảnh
    Returns:
        pd.DataFrame: chứa thông tin các ảnh
    """
    records = []
    classes = sorted(
        [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]
    )
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        images = sorted(os.listdir(cls_path))
        if max_images_per_class:
            images = images[:max_images_per_class]
        for fname in images:
            fpath = os.path.join(cls_path, fname)
            try:
                with Image.open(fpath) as im:
                    w, h = im.size
                    channels = len(im.getbands())
                    mode = im.mode
                    filesize = os.path.getsize(fpath)
                    aspect = round(w / h, 3) if h != 0 else None
                    records.append(
                        {
                            "filepath": fpath,
                            "label": cls,
                            "filename": fname,
                            "width": w,
                            "height": h,
                            "aspect": aspect,
                            "channels": channels,
                            "mode": mode,
                            "filesize_bytes": filesize,
                        }
                    )
            except Exception as e:
                print(f"Warning: cannot open {fpath}: {e}")
    df = pd.DataFrame.from_records(records)
    return df
