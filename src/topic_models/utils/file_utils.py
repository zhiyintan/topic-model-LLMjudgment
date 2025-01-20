import os
import pandas as pd
import numpy as np
from pathlib import Path


def get_file_name_list(path):
    """
    Get a list of file names in a directory.

    Args:
        path (str): Path to the directory.

    Returns:
        list: List of file names in the directory.
    """
    return [file_name for file_name in os.listdir(path)]

import os
import pandas as pd
from pathlib import Path


def create_path(paths):
    """
    Create directories for a list of paths if they do not exist.

    Args:
        paths (list): A list of directory paths to create.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_tsv(df, path):
    """
    Save a DataFrame as a tab-separated file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): The file path where the DataFrame will be saved.
    """
    df.to_csv(path, sep='\t', encoding='utf-8', index=False)


def create_results_file(file_path, header):
    """
    Create a results file with a specified header.

    Args:
        file_path (str): The path of the file to create.
        header (str): The header to write into the file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.isfile(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(header)


def get_file_dir(path):
    """
    Retrieve all file paths ending with '.csv' in a directory.

    Args:
        path (str): The directory path to search.

    Returns:
        list: List of file paths ending with '.csv'.
    """
    file_path_list = []
    for file_name in os.listdir(path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(path, file_name)
            file_path_list.append(file_path)
    return file_path_list


def save_evaluation_metrics(file_path, test_file, metrics, mode='a'):
    """
    Save evaluation metrics to a CSV file in append mode.

    Args:
        file_path (str): Path to the output file.
        test_file (str): Name of the test file evaluated.
        metrics (dict): Dictionary of evaluation metrics.
        mode (str): File mode, default is append ('a').
    """
    # Prepare the data row
    data_row = {
        "test_file": test_file,
        "llm_model": test_file.split('_')[0],
        "number of topics": 100 if int(test_file.split('_')[1]) > 80 else 50,
        "iteration": test_file.split('_')[-1].rsplit('.')[0],
        "mean_precision_at_k": metrics['mean_precision_at_k'],
        "mean_recall_at_k": metrics['mean_recall_at_k'],
        "mean_average_precision": metrics['mean_average_precision'],
        "mean_ndcg_at_k": metrics['mean_ndcg_at_k']
    }

    # Check if file exists to add header or not
    write_header = not os.path.exists(file_path)

    # Write to the file
    with open(file_path, mode, encoding='utf-8') as f:
        # Write header if file doesn't exist
        if write_header:
            f.write("test_file\tllm_model\tnumber of topics\titeration\tmean_precision_at_k\tmean_recall_at_k\tmean_average_precision\tmean_ndcg_at_k\n")
        # Write the data row
        f.write(f"{data_row['test_file']}\t"
                f"{data_row['llm_model']}\t{data_row['number of topics']}\t{data_row['iteration']}\t"
                f"{data_row['mean_precision_at_k']:.4f}\t{data_row['mean_recall_at_k']:.4f}\t"
                f"{data_row['mean_average_precision']:.4f}\t{data_row['mean_ndcg_at_k']:.4f}\n")
