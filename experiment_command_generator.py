import itertools
import os
import stat
from datetime import datetime


def generate_experiment_command(module, num_layers, hidden_dim, emb_epochs, sup_epochs, gan_epochs, embedding_dropout,
                    recovery_dropout, supervisor_dropout, generator_dropout, discriminator_dropout, learning_rate,
                                trace, seq_len, batch_size, scaling_method, recovery_sigmoid, device):
    experiment_save_dir = f'~/experiments/timegan-pytorch/alibaba2018/3_layers_tests/num_layers-{num_layers}-hidden_dim-{hidden_dim}-batch_size-{batch_size}-recover_sigmoid-{recovery_sigmoid}'
    return f'python main.py --device {device} --module {module} --num_layers {num_layers} --hidden_dim {hidden_dim} --emb_epochs {emb_epochs} ' \
           f'--sup_epochs {sup_epochs} --gan_epochs {gan_epochs} --embedding_dropout {embedding_dropout} --recovery_dropout {recovery_dropout} ' \
           f'--supervisor_dropout {supervisor_dropout} --generator_dropout {generator_dropout} --discriminator_dropout {discriminator_dropout} --learning_rate {learning_rate} ' \
           f'--trace {trace} --seq_len {seq_len} --batch_size {batch_size} --n_samples 10 --scaling_method {scaling_method} ' \
           f'--recovery_sigmoid {recovery_sigmoid} --experiment_save_dir {experiment_save_dir}\n'

modules = ['gru']
num_layers = [3]
hidden_dims = [8, 10, 12, 14, 16]
emb_epochs = [50]
sup_epochs = [50]
gan_epochs = [200]
embedding_dropout = [0]
recovery_dropout = [0]
supervisor_dropout = [0]
generator_dropout = [0]
discriminator_dropout = [0]
recovery_sigmoid = ['true']

learning_rates = [1e-3]

traces = ['alibaba2018']
seq_lengths = [288]
batch_sizes = [32,64,128]
scaling_methods = ['minmax']  # 'minmax'
device = 'cuda'

parameterization = [modules, num_layers, hidden_dims, emb_epochs, sup_epochs, gan_epochs, embedding_dropout,
                    recovery_dropout, supervisor_dropout, generator_dropout, discriminator_dropout,
                    learning_rates, traces, seq_lengths, batch_sizes, scaling_methods, recovery_sigmoid]

parameterization_combinations = list(itertools.product(*parameterization))

commands = [
    generate_experiment_command(module, num_layers, hidden_dim, emb_epochs, sup_epochs, gan_epochs, embedding_dropout,
                                recovery_dropout, supervisor_dropout, generator_dropout, discriminator_dropout,
                                learning_rate, trace, seq_len, batch_size, scaling_method, recovery_sigmoid, device) for
    module, num_layers, hidden_dim, emb_epochs, sup_epochs, gan_epochs, embedding_dropout, recovery_dropout, supervisor_dropout, generator_dropout, discriminator_dropout, learning_rate, trace, seq_len, batch_size, scaling_method, recovery_sigmoid
    in parameterization_combinations]

sh_filename = "3-layers-testing-" + datetime.now().strftime("%j-%Y-%H-%M") + ".sh"
sh_file = open(sh_filename, "w")
sh_file.writelines(commands)
st = os.stat(sh_filename)
os.chmod(sh_filename, st.st_mode | stat.S_IEXEC)
