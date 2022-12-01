import itertools
import os
import stat
from datetime import datetime


def generate_experiment_command(module, num_layers, hidden_dim, emb_epochs, sup_epochs, gan_epochs, embedding_dropout,
                    recovery_dropout, supervisor_dropout, generator_dropout, discriminator_dropout, learning_rate,
                                trace, seq_len, batch_size, scaling_method):
    experiment_save_dir = '~/experiments/timegan-pytorch/alibaba2018/small-tests/test/tensorboard/num_layers-{num_layers}-hidden_dim-{hidden_dim}-gan_epochs-{gan_epochs}-batch_size-{batch_size}'
    return f'python main.py --module {module} --num_layers {num_layers} --hidden_dim {hidden_dim} --emb_epochs {emb_epochs} ' \
           f'--sup_epochs {sup_epochs} --gan_epochs {gan_epochs} --embedding_dropout {embedding_dropout} --recovery_dropout {recovery_dropout} ' \
           f'--supervisor_dropout {supervisor_dropout} --generator_dropout {generator_dropout} --discriminator_dropout {discriminator_dropout} --learning_rate {learning_rate} ' \
           f'--trace {trace} --seq_len {seq_len} --n_samples 10 --scaling_method {scaling_method} ' \
           f'--experiment_save_dir {experiment_save_dir}' \
           f'\n'

modules = ['gru']
num_layers = [2]
hidden_dims = [5]
emb_epochs = [5]
sup_epochs = [5]
gan_epochs = [50]
embedding_dropout = [0.5]
recovery_dropout = [0.5]
supervisor_dropout = [0.5]
generator_dropout = [0.5]
discriminator_dropout = [0.5]

learning_rates = [1e-3]

traces = ['alibaba2018']
seq_lengths = [288]
batch_sizes = [32,64,128]
scaling_methods = ['minmax']  # 'minmax'

parameterization = [modules, num_layers, hidden_dims, emb_epochs, sup_epochs, gan_epochs, embedding_dropout,
                    recovery_dropout, supervisor_dropout, generator_dropout, discriminator_dropout,
                    learning_rates, traces, seq_lengths, batch_sizes, scaling_methods]

parameterization_combinations = list(itertools.product(*parameterization))

commands = [
    generate_experiment_command(module, num_layers, hidden_dim, emb_epochs, sup_epochs, gan_epochs, embedding_dropout,
                                recovery_dropout, supervisor_dropout, generator_dropout, discriminator_dropout,
                                learning_rate, trace, seq_len, batch_size, scaling_method) for
    module, num_layers, hidden_dim, emb_epochs, sup_epochs, gan_epochs, embedding_dropout, recovery_dropout, supervisor_dropout, generator_dropout, discriminator_dropout, learning_rate, trace, seq_len, batch_size, scaling_method
    in parameterization_combinations]

sh_filename = "pytorch-testing-" + datetime.now().strftime("%j-%Y-%H-%M") + ".sh"
sh_file = open(sh_filename, "w")
sh_file.writelines(commands)
st = os.stat(sh_filename)
os.chmod(sh_filename, st.st_mode | stat.S_IEXEC)
