import pandas as pd
from tqdm import tqdm


def fiter_dataset(
        dataframe: pd.DataFrame,
        labels_item
) -> pd.DataFrame:
    # Errors 'coerce' converts invalid numbers to NaN
    dataframe["start_seconds"] = pd.to_numeric(
        dataframe["start_seconds"], errors="coerce"
    )
    dataframe["end_seconds"] = pd.to_numeric(
        dataframe["end_seconds"], errors="coerce"
    )
    dataframe.dropna(subset=dataframe.columns, inplace=True)
    loader_dividing = tqdm(labels_item)
    for label_name, mid in loader_dividing:
        dataframe = dataframe[
            dataframe["positive_labels"].str.contains(
                mid, regex=False, na=False
            )
        ]
    return dataframe
