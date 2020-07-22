from torch import nn
import brevitas.nn as qnn
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import os


def plot_kernels(model, text="", output=None):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, qnn.QuantLinear):
            weights = module.weight
            weights = weights.reshape(-1).detach().cpu().numpy()
            # print(module.bias)
            # math.log(abs(weights))
            plt.hist(abs(weights), bins=10 ** np.linspace(np.log10(0.0001), np.log10(2.5), 100), alpha=0.6, label=name)
            # plt.label
    plt.legend(loc='upper left')
    plt.xscale('log')
    if model.quantized_model:
        precision = model.weight_precision
        plt.title("Quant (" + str(precision) + "b) Model Weights " + text)
    else:
        plt.title("Float Model Weights " + text)
    now = datetime.now()
    time = now.strftime("%d-%m-%Y_%H-%M-%S")
    plt.ylabel("Number of Weights")
    plt.xlabel("Absolute Weights")
    if output is None:
        output = os.path.join('weight_dists/', ('weight_dist_' + str(time) + '.png'))
    plt.savefig(output)
    plt.show()
