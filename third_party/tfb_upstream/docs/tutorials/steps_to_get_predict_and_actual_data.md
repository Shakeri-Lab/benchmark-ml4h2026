## Steps to get the model's predicted values and the target values

1. If you want to save the model's predicted values and the target values, you should set `--save-true-pred` to `True` when running the benchmark.
Such as:
```shell
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 256, "lr": 0.01, "horizon": 60, "seq_len": 104}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "your_result_path" --save-true-pred True
```
2. After the benchmark finishes running, the predicted values and target values are saved in the result file. For example, if you set `--save-path "your_result_path"`, the result file will be located at `/result/your_result_path`.
3. The result file is in `.tar.gz` format. You need to extract this file to access the results. 
4. The predicted values are stored in the `"inference_data"` column, while the target values are in the `"actual_data"` column. However, you may notice that they appear as garbled code because they are encoded in base64. You can use the function below to get the decoded predicted and target values.

```python
import base64
import os
import pickle
import time

import numpy as np
import pandas as pd


def to_csv(data, save_dir: str, save_name: str):
    """
    Save the input data (either a DataFrame or a 3D NumPy array) into CSV file(s).

    If the input is a pandas DataFrame, it will be saved directly to the given directory.
    If the input is a 3D NumPy array (with shape [num, time, dim]), each 2D slice (data[i])
    will be saved into a separate subdirectory named 'sample_i'.

    :param data: The data to save, either a pandas DataFrame or a NumPy array of shape (num, time, dim).
    :param save_dir: The root directory where the files will be saved.
    :param save_name: The name of the CSV file(s) to be written.
    :raises TypeError: If the input data type is not supported.
    """
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        data.to_csv(os.path.join(save_dir, save_name), index=False)

    elif isinstance(data, np.ndarray):
        num = data.shape[0]
        for i in range(num):
            sample_dir = os.path.join(save_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            df = pd.DataFrame(data[i])
            df.to_csv(os.path.join(sample_dir, save_name), index=False)

    elif isinstance(data, list):
        num = len(data)
        for i in range(num):
            sample_dir = os.path.join(save_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            df = data[i]
            df.to_csv(os.path.join(sample_dir, save_name), index=False)

    else:
        raise TypeError("Unsupported type for data. Must be pd.DataFrame or np.ndarray.")


def decode_data(filepath: str):
    """
    Load the result CSV file and decode the base64-encoded 'inference_data' and 'actual_data' columns.

    :param filepath: Path to the input CSV file containing encoded data.
    :return: None. The decoded data will be saved as CSV files in corresponding folders.
    """
    data = pd.read_csv(filepath)  # Read the CSV file with encoded columns

    for index, row in data.iterrows():
        # Decode base64 strings and deserialize them back to original DataFrames
        decoded_inference_data = base64.b64decode(row["inference_data"])
        decoded_actual_data = base64.b64decode(row["actual_data"])
        inference_data = pickle.loads(decoded_inference_data)
        actual_data = pickle.loads(decoded_actual_data)

        # Construct directory name by removing special characters from model parameters
        file_name = os.path.splitext(row["file_name"])[0]
        model_name = row["model_name"]
        model_params = row["model_params"].translate(str.maketrans('', '', '":, {}'))
        save_dir = f"{file_name}_{model_name}_{model_params}"
        timestamp = f"{int(time.time() * 1000)}"
        save_dir = os.path.join(save_dir, timestamp)
        print(f"Saving data to directory: {save_dir}")

        # Save the decoded data as CSV files
        to_csv(inference_data, save_dir, "inference_data.csv")
        to_csv(actual_data, save_dir, "actual_data.csv")


# Example usage
your_result_csv_path = r"/path/to/your_result.csv"
decode_data(your_result_csv_path)
```
The decoded results will be saved in the following directory structure:
```
{dataset}_{model_name}_{model_params}
└── {timestamp}
    ├── sample_0
    │   ├── inference_data.csv
    │   └── actual_data.csv
    ├── sample_1
    │   ├── inference_data.csv
    │   └── actual_data.csv
    └── ...
```
