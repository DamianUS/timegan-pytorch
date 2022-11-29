# -*- coding: UTF-8 -*-
# Local modules
import os
import pickle
import copy
from typing import Dict, Union

# 3rd party modules
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.tensorboard import SummaryWriter

# Self-written modules
from models.dataset import TimeGANDataset

def embedding_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    e_opt: torch.optim.Optimizer, 
    r_opt: torch.optim.Optimizer, 
    args: Dict, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None
) -> None:
    """The training loop for the embedding and recovery functions
    """  
    logger = trange(args.emb_epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:   
        for X_mb, T_mb in tqdm(dataloader, desc='Intra-epochs iteration', colour='yellow', leave=False):
            if X_mb.shape[0] == args.batch_size:
                # Reset gradients
                model.zero_grad()

                # Forward Pass
                # time = [args.max_seq_len for _ in range(len(T_mb))]
                _, E_loss0, E_loss_T0 = model(X=X_mb, T=T_mb, Z=None, obj="autoencoder")
                loss = np.sqrt(E_loss_T0.item())

                # Backward Pass
                E_loss0.backward()

                # Update model parameters
                e_opt.step()
                r_opt.step()

        # Log loss for final batch of each epoch (29 iters)
        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
        if writer:
            writer.add_scalar(
                "Embedding/Loss:", 
                loss, 
                epoch
            )
            writer.flush()

def supervisor_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    s_opt: torch.optim.Optimizer, 
    g_opt: torch.optim.Optimizer, 
    args: Dict, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None
) -> None:
    """The training loop for the supervisor function
    """
    logger = trange(args.sup_epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        for X_mb, T_mb in tqdm(dataloader, desc='Intra-epochs iteration', colour='yellow', leave=False):
            if X_mb.shape[0] == args.batch_size:
                # Reset gradients
                model.zero_grad()

                # Forward Pass
                S_loss = model(X=X_mb, T=T_mb, Z=None, obj="supervisor")

                # Backward Pass
                S_loss.backward()
                loss = np.sqrt(S_loss.item())

                # Update model parameters
                s_opt.step()

        # Log loss for final batch of each epoch (29 iters)
        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
        if writer:
            writer.add_scalar(
                "Supervisor/Loss:", 
                loss, 
                epoch
            )
            writer.flush()

def joint_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    e_opt: torch.optim.Optimizer, 
    r_opt: torch.optim.Optimizer, 
    s_opt: torch.optim.Optimizer, 
    g_opt: torch.optim.Optimizer, 
    d_opt: torch.optim.Optimizer, 
    args: Dict, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None, 
) -> None:
    """The training loop for training the model altogether
    """
    logger = trange(
        args.gan_epochs,
        desc=f"Epoch: 0, E_loss: 0, G_loss: 0, D_loss: 0"
    )
    
    for epoch in logger:
        intra_epoch_progress_bar = tqdm(dataloader, desc=f'Intra-epoch: 0, E_loss: 0, G_loss: 0, D_loss: 0"', colour='yellow', leave=False)
        for X_mb, T_mb in intra_epoch_progress_bar:
            if X_mb.shape[0] == args.batch_size:
                ## Generator Training
                for _ in range(2):
                    # Random Generator
                    #Z_mb = torch.rand((args.batch_size, args.max_seq_len, args.Z_dim))
                    Z_mb = list()
                    for i in range(args.batch_size):
                        temp = np.zeros([args.max_seq_len, args.Z_dim])
                        temp_Z = np.random.uniform(0., 1, [T_mb[i], args.Z_dim])
                        temp[:T_mb[i], :] = temp_Z
                        Z_mb.append(temp_Z)
                    # Forward Pass (Generator)
                    model.zero_grad()
                    G_loss = model(X=X_mb, T=T_mb, Z=Z_mb, obj="generator")
                    G_loss.backward()
                    G_loss = G_loss.item()

                    # Update model parameters
                    g_opt.step()
                    s_opt.step()

                    # Forward Pass (Embedding)
                    model.zero_grad()
                    E_loss, _, E_loss_T0 = model(X=X_mb, T=T_mb, Z=Z_mb, obj="autoencoder")
                    E_loss.backward()
                    E_loss = np.sqrt(E_loss.item())

                    # Update model parameters
                    e_opt.step()
                    r_opt.step()

                # Random Generator
                #Z_mb = torch.rand((args.batch_size, args.max_seq_len, args.Z_dim))
                Z_mb = list()
                for i in range(args.batch_size):
                    temp = np.zeros([args.max_seq_len, args.Z_dim])
                    temp_Z = np.random.uniform(0., 1, [T_mb[i], args.Z_dim])
                    temp[:T_mb[i], :] = temp_Z
                    Z_mb.append(temp_Z)

                ## Discriminator Training
                model.zero_grad()
                # Forward Pass
                D_loss = model(X=X_mb, T=T_mb, Z=Z_mb, obj="discriminator")

                # Check Discriminator loss
                if D_loss > args.dis_thresh:
                    # Backward Pass
                    D_loss.backward()

                    # Update model parameters
                    d_opt.step()
                D_loss = D_loss.item()
                intra_epoch_progress_bar.set_description(
                    f"Minibatch: {epoch}, E: {E_loss:.4f}, G: {G_loss:.4f}, D: {D_loss:.4f}"
                )

        logger.set_description(
            f"Epoch: {epoch}, E: {E_loss:.4f}, G: {G_loss:.4f}, D: {D_loss:.4f}"
        )
        if writer:
            writer.add_scalar(
                'Joint/Embedding_Loss:', 
                E_loss, 
                epoch
            )
            writer.add_scalar(
                'Joint/Generator_Loss:', 
                G_loss, 
                epoch
            )
            writer.add_scalar(
                'Joint/Discriminator_Loss:', 
                D_loss, 
                epoch
            )
            writer.flush()
        save_epoch(args, epoch, model)

def timegan_trainer(model, data, time, args):
    """The training procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model model that generates synthetic data
        - data (numpy.ndarray): The data for training the model
        - time (numpy.ndarray): The time for the model to be conditioned on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """

    # Initialize TimeGAN dataset and dataloader
    dataset = TimeGANDataset(data, time)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    model.to(args.device)

    # Initialize Optimizers
    e_opt = torch.optim.Adam(model.embedder.parameters(), lr=args.learning_rate)
    r_opt = torch.optim.Adam(model.recovery.parameters(), lr=args.learning_rate)
    s_opt = torch.optim.Adam(model.supervisor.parameters(), lr=args.learning_rate)
    g_opt = torch.optim.Adam(model.generator.parameters(), lr=args.learning_rate)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=args.learning_rate)
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(f'{args.experiment_save_dir}/tensorboard'))

    print("\nStart Embedding Network Training")
    embedding_trainer(
        model=model, 
        dataloader=dataloader, 
        e_opt=e_opt, 
        r_opt=r_opt, 
        args=args, 
        writer=writer
    )

    print("\nStart Training with Supervised Loss Only")
    supervisor_trainer(
        model=model,
        dataloader=dataloader,
        s_opt=s_opt,
        g_opt=g_opt,
        args=args,
        writer=writer
    )

    print("\nStart Joint Training")
    joint_trainer(
        model=model,
        dataloader=dataloader,
        e_opt=e_opt,
        r_opt=r_opt,
        s_opt=s_opt,
        g_opt=g_opt,
        d_opt=d_opt,
        args=args,
        writer=writer,
    )

    # Save model, args, and hyperparameters
    torch.save(args, f"{args.model_path}/args.pickle")
    torch.save(model.state_dict(), f"{args.model_path}/model.pt")
    print(f"\nSaved at path: {args.model_path}")

def timegan_generator(model, T, args):
    """The inference procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model model that generates synthetic data
        - T (List[int]): The time to be generated on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """
    # Load model for inference
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model directory not found...")

    # # Load arguments and model
    # with open(f"{args.model_path}/args.pickle", "rb") as fb:
    #     args = torch.load(fb)
    
    model.load_state_dict(torch.load(f"{args.model_path}/model.pt"))
    
    #print("\nGenerating Data...")
    # Initialize model to evaluation mode and run without gradients
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        # Generate fake data
        Z = torch.rand((len(T), args.max_seq_len, args.Z_dim))
        
        generated_data = model(X=None, T=T, Z=Z, obj="inference")

    return generated_data.numpy()

def save_generated_data(generated_data, scaler, experiment_save_dir, n_samples=10):
    if not os.path.exists(experiment_save_dir):
        os.makedirs(experiment_save_dir, exist_ok=True)
    generated_data_n_samples = generated_data[:n_samples]
    generated_data_rescaled = np.reshape(scaler.inverse_transform(generated_data_n_samples.reshape(-1, 1)),
                                         generated_data_n_samples.shape)
    for i, generated_sample in enumerate(generated_data_rescaled):
        np.savetxt(f'{experiment_save_dir}/sample_{i}.csv', generated_sample, delimiter=",", fmt='%f')


def save_epoch(args, epoch, model):
    args_copy = copy.deep_copy(args)
    args_copy.model_path = f'{args_copy.model_path}/epoch_{epoch}'
    if not os.path.exists(args_copy.model_path):
        os.makedirs(args_copy.model_path , exist_ok=True)
    torch.save(args_copy, f'{args_copy.model_path}/args.pickle')
    torch.save(model.state_dict(), f'{args_copy.model_path}/model.pt')
    print(f"\nSaved epoch_{epoch} at path: {args_copy.model_path}")