from torch import nn
import brevitas.nn as qnn
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import os


def plot_kernels(model, text="", output=None):
    weight_plt = plt.figure()
    weight_ax = weight_plt.add_subplot()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, qnn.QuantLinear):
            weights = module.weight
            weights = weights.reshape(-1).detach().cpu().numpy()
            weight_ax.hist(abs(weights), bins=10 ** np.linspace(np.log10(0.0001), np.log10(2.5), 100), alpha=0.6, label=name)
    weight_ax.legend(loc='upper left')
    weight_ax.set_xscale('log')
    if model.quantized_model:
        precision = model.weight_precision
        weight_ax.set_title("Quant (" + str(precision) + "b) Model Weights " + text)
    else:
        weight_ax.set_title("Float Model Weights " + text)
    now = datetime.now()
    time = now.strftime("%d-%m-%Y_%H-%M-%S")
    weight_ax.set_ylabel("Number of Weights")
    weight_ax.set_xlabel("Absolute Weights")
    if output is None:
        os.makedirs('weight_dists/',exist_ok=True)
        output = os.path.join('weight_dists/', ('weight_dist_' + str(time) + '.png'))
    weight_plt.savefig(output)
    weight_plt.show()
