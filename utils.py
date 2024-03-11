import tensorflow as tf
import numpy as np
import csv
import os
        
from scipy.io import wavfile
from scipy import signal


# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
              class_names.append(row['display_name'])
    return class_names

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                   original_sample_rate * desired_sample_rate))
        waveform = signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

def normalize_matrix(matrix):
    
    # Normalize each column
    matrix = [normalize_to_sum_one(column) for column in matrix]
    
    return np.array(matrix)

def normalize_to_sum_one(values):
    total = sum(values)
    normalized_values = [value / total for value in values]
    return normalized_values

def time_format(seconds: float) -> str:
    if seconds < 0:
        seconds = -seconds
        neg = '-'
    else:
        neg = ''
    if seconds is not None:
        ms = round((seconds%1)*1000)
        seconds = int(seconds)
        d = seconds // (3600 * 24)
        h = seconds // 3600 % 24
        m = seconds % 3600 // 60
        s = seconds % 3600 % 60
        #print(ms)
        if d > 0:
            return neg+'{:02d}:{:02d}:{:02d}:{:02d}.{:03d}'.format(d, h, m, s, ms)
        elif h > 0:
            return neg+'{:02d}:{:02d}:{:02d}.{:03d}'.format(h, m, s, ms)
        elif m > 0:
            return neg+'{:02d}:{:02d}.{:03d}'.format(m, s, ms)
        elif s >= 0:
            return neg+'{:02d}:{:02d}.{:03d}'.format(m, s, ms)
    return '-'
def seconds(time_format: str) -> float:
    return sum(x * int(t) for x, t in zip([1, 60, 3600, 24*3600], time_format.split(":")[::-1]))


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

