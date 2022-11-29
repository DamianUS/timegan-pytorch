import argparse
from distutils import util
import os
from models.utils import timegan_generator, save_generated_data
import pickle
import torch
from models.timegan import TimeGAN
import data_load
from tqdm import tqdm
from natsort import natsorted


def main(args):
    print(args)
    experiment_directories = []
    root_dir = f'{args.experiment_directory_path}/model'
    if args.recursive == True:
        experiment_directories = []
        for subdir, dirs, files in os.walk(args.experiment_directory_path):
            if 'epoch' in subdir:
                experiment_directories.append(subdir)
        experiment_directories = natsorted(experiment_directories)
        for experiment_directory in experiment_directories:
            print(experiment_directory)
    elif args.epoch >= 0:
        experiment_directories.append(f'{root_dir}/epoch_{args.epoch}')
    else:
        experiment_directories.append(root_dir)
    progress_bar = tqdm(experiment_directories)
    for experiment_dir in progress_bar:
        progress_bar.set_description(f'Generating samples with model at {os.path.basename(os.path.normpath(experiment_dir))}')
        print(experiment_dir)
        with open(f"{experiment_dir}/args.pickle", "rb") as fb:
            recovered_args = torch.load(fb)

        recovered_args.experiment_save_dir = f'{experiment_dir}'
        recovered_args.is_train = False
        recovered_args.n_samples = args.n_samples_export
        recovered_args.max_seq_len = recovered_args.seq_len
        recovered_args.model_path = experiment_dir
        print("recovered_args modified", recovered_args)

        model = TimeGAN(recovered_args)
        if recovered_args.ori_data_filename is not None:
            X, T, scaler = data_load.get_dataset(ori_data_filename=recovered_args.ori_data_filename, sequence_length=recovered_args.seq_len,
                                                 stride=1, trace_timestep=recovered_args.trace_timestep, shuffle=False, seed=13,
                                                 scaling_method='standard')
        else:
            X, T, scaler = data_load.get_datacentertraces_dataset(trace=recovered_args.trace, trace_type=recovered_args.trace_type,
                                                                  sequence_length=recovered_args.seq_len, stride=1,
                                                                  trace_timestep=recovered_args.trace_timestep, shuffle=False,
                                                                  seed=13, scaling_method='standard')
        generated_data = timegan_generator(model, T, recovered_args)
        save_generated_data(generated_data, scaler, f'{experiment_dir}/generated_data', n_samples=10)


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_directory_path',
        type=str)
    parser.add_argument(
        '--epoch',
        default=-1,
        type=int)
    parser.add_argument(
        '--n_samples_export',
        default=10,
        type=int)
    parser.add_argument(
        '--recursive',
        default=False,
        type=lambda x: bool(util.strtobool(str(x))))
    args = parser.parse_args()
    main(args)