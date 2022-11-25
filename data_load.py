import numpy as np
from datacentertracesdatasets import loadtraces
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def get_datacentertraces_dataset(trace='alibaba2018', trace_type='machine_usage', sequence_length=60, stride=1, trace_timestep=300, shuffle=False, seed=13, scaling_method='standard'):
    np.random.seed(seed)
    original_df = loadtraces.get_trace(trace, trace_type=trace_type, stride_seconds=trace_timestep)
    start_sequence_range = list(range(0, original_df.shape[0] - sequence_length, stride))
    if shuffle == True:
        np.random.shuffle(start_sequence_range)
    splitted_original_x = np.array([original_df[start_index:start_index+sequence_length] for start_index in start_sequence_range])
    scaled_ori_data, scaler, scaler_params  = __scale_data(splitted_original_x, scaling_method=scaling_method)
    T = np.full((splitted_original_x.shape[0]), splitted_original_x.shape[1])
    return scaled_ori_data, T, scaler

def get_dataset(ori_data_filename=None, sequence_length=60, stride=1, trace_timestep=300, shuffle=False, seed=13, scaling_method='standard'):
    np.random.seed(seed)
    original_df = pd.read_csv(ori_data_filename, header=None)
    start_sequence_range = list(range(0, original_df.shape[0] - sequence_length, stride))
    if shuffle == True:
        np.random.shuffle(start_sequence_range)
    splitted_original_x = np.array([original_df[start_index:start_index + sequence_length] for start_index in start_sequence_range])
    scaled_ori_data, scaler, scaler_params = __scale_data(splitted_original_x, scaling_method=scaling_method)
    T = np.full((splitted_original_x.shape[0]), splitted_original_x.shape[1])
    return scaled_ori_data, T, scaler


def __scale_data(ori_data, scaling_method='standard'):
    assert scaling_method in ['standard', 'minmax'], 'Only standard and minmax scalers are currently supported'
    reshaped_ori_data = ori_data.reshape(-1, 1)
    if scaling_method == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit(reshaped_ori_data)
        scaler_params = [scaler.data_min_, scaler.data_max_]

    elif scaling_method == 'standard':
        scaler = StandardScaler()
        scaler.fit(reshaped_ori_data)
        scaler_params = [scaler.mean_, scaler.var_]
    scaled_ori_data = scaler.transform(reshaped_ori_data).reshape(ori_data.shape)
    return scaled_ori_data, scaler, scaler_params