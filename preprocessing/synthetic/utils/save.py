import numpy as np
import pandas as pd
from datetime import datetime
from math import ceil, floor
from src.data.dataset import Dataset

def save_dataset(
        dataset: Dataset,
        fn: str
) -> None:
    """
    Returns a csv properly formatted with the right elements.
    """

    batch_size = dataset.batch_size
    dataset_size = dataset.len() * dataset.time_length
    n_patients = ceil(dataset.len() / batch_size)
    pat_ids = [floor(i / (n_patients * dataset.time_length)) for i in range(dataset_size)]
    seq_ids = [floor(i / dataset.time_length) for i in range(dataset_size)]
    dates = [pd.to_datetime(datetime(2026, 1, 1)) for _ in range(dataset_size)]

    data_dict = {
        "pat_id": pat_ids,
        "seq_id": seq_ids,
        "date": dates,
        "cgm": dataset.data["targets"].flatten()
    }

    for i in range(dataset.data["features"].shape[2]):
        data_dict[f"feature_{i}"] = dataset.data["features"][:, :, i].flatten()

    df = pd.DataFrame(data_dict)

    df.to_csv(fn, index=False)

    print("dataset saved")