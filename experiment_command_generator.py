import itertools
import os
import stat
from datetime import datetime


def generate_experiment_command(module, num_layers, hidden_dim, emb_epochs, sup_epochs, gan_epochs, learning_rate,
                                trace, seq_len, batch_size, scaling_method):
    experiment_save_dir = f'./experiments/timegan-pytorch/alibaba2018/ecms-experiments-pytorch/num-layers-{num_layers}-hidden_dim-{hidden_dim}-batch_size-{batch_size}'
    return f'python main.py --module {module} --num_layers {num_layers} --hidden_dim {hidden_dim} --emb_epochs {emb_epochs} ' \
           f'--sup_epochs {sup_epochs} --gan_epochs {gan_epochs} --learning_rate {learning_rate} ' \
           f'--trace {trace} --seq_len {seq_len} --batch_size {batch_size} --n_samples 10 --scaling_method {scaling_method} ' \
           f'--experiment_save_dir {experiment_save_dir}' \
           f'\n'

modules = ['gru']
num_layers = [2,3,4]
hidden_dims = [5,10,15]
emb_epochs = [200]
sup_epochs = [200]
gan_epochs = [200]
learning_rates = [1e-3]

traces = ['alibaba2018']
seq_lengths = [288]
batch_sizes = [32,64,128]
scaling_methods = ['minmax']  # 'minmax'

parameterization = [modules, num_layers, hidden_dims, emb_epochs, sup_epochs, gan_epochs, learning_rates, traces,
                    seq_lengths, batch_sizes, scaling_methods]

parameterization_combinations = list(itertools.product(*parameterization))

commands = [
    generate_experiment_command(module, num_layers, hidden_dim, emb_epochs, sup_epochs, gan_epochs, learning_rate,
                                trace, seq_len, batch_size, scaling_method) for
    module, num_layers, hidden_dim, emb_epochs, sup_epochs, gan_epochs, learning_rate, trace, seq_len, batch_size, scaling_method
    in parameterization_combinations]

sh_filename = "ecms-experiments-" + datetime.now().strftime("%j-%Y-%H-%M") + ".sh"
sh_file = open(sh_filename, "w")
sh_file.writelines(commands)
st = os.stat(sh_filename)
os.chmod(sh_filename, st.st_mode | stat.S_IEXEC)
