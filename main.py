# -*- coding: UTF-8 -*-
# Local modules
import argparse
import logging
import os
import pickle
import random
import shutil
import time
from sklearn.preprocessing import StandardScaler

# 3rd-Party Modules
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split
import data_load

# Self-Written Modules
from data.data_preprocess import data_preprocess
from metrics.metric_utils import (
    feature_prediction, one_step_ahead_prediction, reidentify_score
)

from models.timegan import TimeGAN
from models.utils import timegan_trainer, timegan_generator, save_generated_data


def main(args):
    ##############################################
    # Initialize output directories
    ##############################################

    ## Output directories
    # experiment_save_dir = os.path.abspath(f'{args.experiment_save_dir}/generated_data')
    # if not os.path.exists(experiment_save_dir):
    #     os.makedirs(experiment_save_dir, exist_ok=True)

    # TensorBoard directory
    tensorboard_path = os.path.abspath(f'{args.experiment_save_dir}/tensorboard')
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path, exist_ok=True)

    # Model directory
    args.model_path = os.path.abspath(f'{args.experiment_save_dir}/model')
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path, exist_ok=True)

    # Save parameterization
    text_file = open(f'{args.experiment_save_dir}/parameters.txt', "w")
    text_file.write(str(args))
    text_file.close()

    ##############################################
    # Initialize random seed and CUDA
    ##############################################

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        print("Using CUDA\n")
        args.device = torch.device("cuda:0")
        # torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Using CPU\n")
        args.device = torch.device("cpu")
    #########################
    # Load and preprocess data for model
    #########################

    # data_path = "data/stock.csv"
    # X, T, _, args.max_seq_len, args.padding_value = data_preprocess(
    #     data_path, args.max_seq_len
    # )

    if args.ori_data_filename is not None:
        X, T, scaler = data_load.get_dataset(ori_data_filename=args.ori_data_filename, sequence_length=args.seq_len,
                                             stride=1, trace_timestep=args.trace_timestep, shuffle=False, seed=13,
                                             scaling_method='standard')
    else:
        X, T, scaler = data_load.get_datacentertraces_dataset(trace=args.trace, trace_type=args.trace_type,
                                                              sequence_length=args.seq_len, stride=1,
                                                              trace_timestep=args.trace_timestep, shuffle=False,
                                                              seed=13, scaling_method='standard')

    #########################
    # Initialize and Run model
    #########################
    args.feature_dim = X.shape[-1]
    args.Z_dim = X.shape[-1]
    args.padding_value = -1.0
    args.max_seq_len = args.seq_len
    # Log start time
    start = time.time()

    model = TimeGAN(args)
    if args.is_train == True:
        timegan_trainer(model, X, T, args)
    # generated_data = timegan_generator(model, T, args)
    # save_generated_data(generated_data=generated_data, scaler=scaler, experiment_save_dir=experiment_save_dir, n_samples=10)

    # Log end time
    end = time.time()

    # print(f"Generated data preview:\n{generated_data_rescaled[:2, -10:, :2]}\n")
    print(f"Model Runtime: {(end - start) / 60} mins\n")
    print(f"Total Runtime: {(time.time() - start) / 60} mins\n")
    #########################
    # Save train and generated data for visualization
    #########################

    # Save splitted data and generated data
    # with open(f"{args.model_path}/train_data.pickle", "wb") as fb:
    #     pickle.dump(train_data, fb)
    # with open(f"{args.model_path}/train_time.pickle", "wb") as fb:
    #     pickle.dump(train_time, fb)
    # with open(f"{args.model_path}/test_data.pickle", "wb") as fb:
    #     pickle.dump(test_data, fb)
    # with open(f"{args.model_path}/test_time.pickle", "wb") as fb:
    #     pickle.dump(test_time, fb)
    # with open(f"{args.model_path}/fake_data.pickle", "wb") as fb:
    #     pickle.dump(generated_data_rescaled, fb)
    # with open(f"{args.model_path}/fake_time.pickle", "wb") as fb:
    #     pickle.dump(generated_time, fb)

    return None


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()

    # Experiment Arguments
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cpu',
        type=str)
    parser.add_argument(
        '--experiment_save_dir',
        default='./experiments',
        type=str)
    parser.add_argument(
        "--is_train",
        type=str2bool,
        default=True)
    parser.add_argument(
        '--seed',
        default=42,
        type=int)

    # Data Arguments
    parser.add_argument(
        '--seq_len',
        default=288,
        type=int)
    parser.add_argument(
        '--n_samples',
        default=10,
        type=int)
    parser.add_argument(
        '--scaling_method',
        default='standard',
        type=str)
    parser.add_argument(
        '--ori_data_filename',
        default=None,
        type=str)
    parser.add_argument(
        '--trace',
        choices=['alibaba2018', 'azure_v2', 'google2019'],
        default='alibaba2018',
        type=str)
    parser.add_argument(
        '--trace_type',
        default='machine_usage',
        type=str)
    parser.add_argument(
        '--trace_timestep',
        default=300,
        type=int)

    # Model Arguments
    parser.add_argument(
        '--module',
        choices=['lstm', 'gru'],
        default='gru',
        type=str)
    parser.add_argument(
        '--emb_epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--sup_epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--gan_epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        default=20,
        type=int)
    parser.add_argument(
        '--num_layers',
        default=3,
        type=int)
    parser.add_argument(
        '--dis_thresh',
        default=0.15,
        type=float)
    parser.add_argument(
        '--optimizer',
        choices=['adam'],
        default='adam',
        type=str)
    parser.add_argument(
        '--learning_rate',
        default=1e-3,
        type=float)
    parser.add_argument(
        '--embedding_dropout',
        default=0.0,
        type=float)
    parser.add_argument(
        '--recovery_dropout',
        default=0.0,
        type=float)
    parser.add_argument(
        '--supervisor_dropout',
        default=0.0,
        type=float)
    parser.add_argument(
        '--generator_dropout',
        default=0.0,
        type=float)
    parser.add_argument(
        '--discriminator_dropout',
        default=0.0,
        type=float)
    parser.add_argument(
        '--noise_threshold',  # 0.0 means no noise
        default=0.0,
        type=float)
    parser.add_argument(
        '--gamma',  # 0.0 means no noise
        default=1,
        type=int)

    args = parser.parse_args()

    # Call main function
    main(args)
