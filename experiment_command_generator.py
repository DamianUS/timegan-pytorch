import itertools
import os
import stat
from datetime import datetime


def compose_training_command(module, num_layers, hidden_dim, emb_sup_epochs, gan_epochs, embedding_dropout,
                             recovery_dropout, supervisor_dropout, generator_dropout, discriminator_dropout,
                             learning_rate, trace, seq_len, batch_size, scaling_method, device):
    experiment_save_dir = f'~/experiments/timegan-pytorch/alibaba2018/GRU-seq_len-{seq_len}/num_layers-{num_layers}-hidden_dim-{hidden_dim}-batch_size-{batch_size}-module-{module}-emb_sup_e-{emb_sup_epochs}'
    training_command = f'python train.py  --module {module} --num_layers {num_layers} --hidden_dim {hidden_dim} --batch_size {batch_size} ' \
           f'--emb_epochs {emb_sup_epochs} --sup_epochs {emb_sup_epochs} --gan_epochs {gan_epochs} ' \
           f'--seq_len {seq_len} --scaling_method {scaling_method} --learning_rate {learning_rate} ' \
           f'--embedding_dropout {embedding_dropout} --recovery_dropout {recovery_dropout} --trace {trace} --supervisor_dropout {supervisor_dropout} --generator_dropout {generator_dropout} --discriminator_dropout {discriminator_dropout}  --device {device} ' \
           f'--experiment_save_dir {experiment_save_dir}\n'
    return training_command, experiment_save_dir

def compose_generation_command (experiment_save_dir, n_samples_export):
    generation_command = f'python generate.py --experiment_directory_path {experiment_save_dir} --n_samples_export {n_samples_export} --recursive true\n'
    return generation_command

def compose_training_and_generation_command(module, num_layers, hidden_dim, emb_sup_epochs, gan_epochs, embedding_dropout,
                             recovery_dropout, supervisor_dropout, generator_dropout, discriminator_dropout,
                             learning_rate, trace, seq_len, batch_size, scaling_method, device, n_samples_export):
    training_command, experiment_save_dir = compose_training_command(module, num_layers, hidden_dim, emb_sup_epochs, gan_epochs, embedding_dropout,
                             recovery_dropout, supervisor_dropout, generator_dropout, discriminator_dropout,
                             learning_rate, trace, seq_len, batch_size, scaling_method, device)
    generation_command = compose_generation_command (experiment_save_dir, n_samples_export)

    return training_command + generation_command

modules = ['gru']
num_layers = [2, 3, 4]
hidden_dims = [16, 32, 64, 128]
emb_sup_epochs = [200, 2000]
gan_epochs = [2000]
embedding_dropout = [0]
recovery_dropout = [0]
supervisor_dropout = [0]
generator_dropout = [0]
discriminator_dropout = [0]

learning_rates = [1e-3]

traces = ['alibaba2018']
seq_lengths = [288]
batch_sizes = [128]
scaling_methods = ['minmax']  # 'minmax'
device = 'cuda'
n_samples_export = 10

parameterization = [modules, num_layers, hidden_dims, emb_sup_epochs, gan_epochs, embedding_dropout,
                    recovery_dropout, supervisor_dropout, generator_dropout, discriminator_dropout,
                    learning_rates, traces, seq_lengths, batch_sizes, scaling_methods]

parameterization_combinations = list(itertools.product(*parameterization))

commands = [
    compose_training_and_generation_command(module, num_layers, hidden_dim, emb_sup_epochs, gan_epochs, embedding_dropout,
                             recovery_dropout, supervisor_dropout, generator_dropout, discriminator_dropout,
                             learning_rate, trace, seq_len, batch_size, scaling_method, device, n_samples_export) for
    module, num_layers, hidden_dim, emb_sup_epochs, gan_epochs, embedding_dropout, recovery_dropout, supervisor_dropout, generator_dropout, discriminator_dropout, learning_rate, trace, seq_len, batch_size, scaling_method
    in parameterization_combinations]

sh_filename = "gru-" + datetime.now().strftime("%j-%Y-%H-%M") + ".sh"
sh_file = open(sh_filename, "w")
sh_file.writelines(commands)
st = os.stat(sh_filename)
os.chmod(sh_filename, st.st_mode | stat.S_IEXEC)
