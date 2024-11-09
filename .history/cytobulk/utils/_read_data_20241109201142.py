import os
from pathlib import Path
import warnings
import pandas as pd

def check_paths(output_folder,output_prefix=None):
    # Create relative path
    output_path = os.path.join(os.getcwd(), output_folder)

    # Make sure that the folder exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if os.path.exists(os.path.join(output_path, f"{output_prefix}assigned_locations.csv")):
        print("\033[91mWARNING\033[0m: Running this will overwrite previous results, choose a new"
              " 'output_folder' or 'output_prefix'")

    return output_path

def read_file(file_path, file_label):
    """
    Read data with given path.

    Parameters
    ----------
        file_path
            file path.
        file_label
            the file label to raise exception.

    Returns
    -------
        the read file.

    """
    try:
        file_delim = "," if file_path.endswith(".csv") else "\t"
        with warnings.catch_warnings():
            file_data = pd.read_csv(file_path, sep=file_delim).dropna()
    except Exception as e:
        raise IOError (f"Make sure you provided the correct path to {file_label} files. "
                    "The following input file formats are supported: .csv with comma ',' as "
                    "delimiter, .txt or .tsv with tab '\\t' as delimiter.")
    return file_data




