import os
import re
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def create_file_mapping(folder_path):
    """
    Creates a dictionary, mapping the first three digits of each file in the folder to its full file path.

    Args:
        folder_path (str): Path to the data folder, which contains the dataset files.

    Returns:
        dict: A dictionary where keys are the first three digits of the filenames and values are the full file paths.
    """
    file_mapping = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            key = file_name[:3]
            file_mapping[key] = os.path.join(folder_path, file_name)
    return file_mapping

def extract_training_split_from_filename(filename):
    """
    Extracts the training split index from a dataset filename.
    The filename is assumed to follow the pattern: <prefix>_<training_end>_<anomaly_start>_<anomaly_end>.txt
    Example: '..._35000_52000_52620.txt' -> returns 35000
    """
    numbers = re.findall(r'\d+', filename)
    if len(numbers) < 3:
        raise ValueError(f"Filename doesn't contain enough numeric fields: {filename}")
    return int(numbers[-3])

def extract_anomaly_range_from_filename(filename):
    """
    Extracts the ground truth anomaly range (start, end) from a dataset filename.
    The filename is assumed to follow the pattern: <prefix>_<training_end>_<anomaly_start>_<anomaly_end>.txt
    Example: '001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt' => returns (52000, 52620)

    Args:
        filename (str): The full filename.

    Returns:
        tuple: (anomaly_start, anomaly_end) as integers.

    Raises:
        ValueError: If the filename does not contain the expected pattern.
    """

    numbers = re.findall(r'\d+', filename)

    if len(numbers) < 3:
        raise ValueError(f"Filename does not contain enough numeric parts to extract anomaly range: {filename}")
    
    anomaly_start = int(numbers[-2])
    anomaly_end = int(numbers[-1])
    return (anomaly_start, anomaly_end)

def mark_anomalies(data_df, anomaly_range):
    """
    Adds a new column 'is_anomaly' to the dataset, marking rows within the anomaly range as 1 and others as 0.

    Args:
        data_df (pd.DataFrame): The dataset containing the time series data.
        anomaly_range (tuple): A tuple (start, end) representing the anomaly range.

    Returns:
        pd.DataFrame: The updated dataset with the 'is_anomaly' column.
    """
    start, end = anomaly_range
    data_df['is_anomaly'] = 0

    if start > end:
        print(f"Warning: Invalid anomaly_range (start > end): {anomaly_range}. Not marking anomalies.")
        return data_df

    try:
        if isinstance(data_df.index, pd.MultiIndex):
            idx_level0 = data_df.index.get_level_values(0)
            
            if pd.api.types.is_numeric_dtype(idx_level0):
                mask = (idx_level0 >= start) & (idx_level0 <= end)
                if len(mask) == len(data_df):
                    data_df.loc[mask, 'is_anomaly'] = 1
                else:
                    print(f"Warning: Length mismatch between MultiIndex level 0 mask and DataFrame in mark_anomalies for range {start}-{end}. Anomalies might not be marked correctly.")
            else:
                print(f"Warning: mark_anomalies received a MultiIndex whose first level is not numeric. Cannot mark anomalies for range {start}-{end} using this level.")
        else:
            mask = (data_df.index >= start) & (data_df.index <= end)
            data_df.loc[mask, 'is_anomaly'] = 1

    except Exception as e:
        print(f"Error during mark_anomalies for range {start}-{end}: {e}. Anomalies might not be marked correctly.")
        
    return data_df

def prepare_dataset(file_path):
    """
    Prepares the dataset by:
    1. Loading the data into a DataFrame, handling potential multi-column rows.
    2. Flattening the data into a single 1D Pandas Series.
    3. Extracting the anomaly range from the file name.
    4. Marking the anomalies in the DataFrame.

    Args:
        file_path (str): The path to the dataset file.

    Returns:
        pd.DataFrame: The prepared DataFrame with 'Value' and 'is_anomaly' columns.
                      The DataFrame will have a simple integer index.
    """

    try:
        raw_data_df = pd.read_csv(file_path, header=None, sep=r'\s+', dtype=float, engine='python')

        series_data = raw_data_df.stack().dropna().reset_index(drop=True)
        
        data_df = series_data.to_frame(name="Value")

    except pd.errors.EmptyDataError:
        print(f"Warning: Empty or unparseable file: {file_path}. Returning empty DataFrame.")
        return pd.DataFrame(columns=["Value", "is_anomaly"])
    

    except Exception as e:
        print(f"Error loading or processing file {file_path}: {e}. Returning empty DataFrame.")
        return pd.DataFrame(columns=["Value", "is_anomaly"])

    if data_df.empty:
        print(f"Warning: No valid data extracted from {file_path} after processing. Returning empty DataFrame.")
        data_df['is_anomaly'] = pd.Series(dtype=int)
        return data_df

    anomaly_range = extract_anomaly_range_from_filename(os.path.basename(file_path))
    data_df = mark_anomalies(data_df, anomaly_range)
    return data_df

def estimate_period(series):
    """
    We can use autocorrelation to esimate the periodicity of the time-series data. Based on the periodicity, we can decide the window size to use in algorithms 
    like the Autoencoder and Time Series Discords. 
    This function estimates the dominant repeating pattern length (or period) of a time series by computing its autocorrelation — a measure of how similar the signal is 
    to itself at different time lags. Peaks in the autocorrelation suggest periodicity, and we use the first significant peak as the estimated period. 
    If no peak is found, we fall back to a safe default like 50.

    TLDR: Estimates the dominant repeating pattern length (or period) of a time series.
    """
    autocorr = np.correlate(series - np.mean(series), series - np.mean(series), mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    peaks, _ = find_peaks(autocorr)
    if len(peaks) > 0:
        for peak in peaks:
            if peak > 1:
                return peak
        return peaks[0] if peaks[0] > 0 else 50
    else:
        return 50