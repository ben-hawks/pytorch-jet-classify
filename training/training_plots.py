from torch import nn
import brevitas.nn as qnn
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import os
import os.path as path

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
    plt.close(weight_plt)

def plot_metric_vs_bitparam(model_set,metric_results_set,bit_params_set,base_metrics_set,metric_text,outputDir='..'):
    # NOTE: Assumes that the first object in the base metrics set is the true base of comparison
    now = datetime.now()
    time = now.strftime("%d-%m-%Y_%H-%M-%S")

    filename = '{}_vs_bitparams'.format(metric_text) + str(time) + '.png'

    rel_perf_plt = plt.figure()
    rel_perf_ax = rel_perf_plt.add_subplot()

    for model, metric_results, bit_params in zip(model_set, metric_results_set, bit_params_set):
        nbits = model.weight_precision if hasattr(model, 'weight_precision') else 32
        rel_perf_ax.plot(bit_params, metric_results, linestyle='solid', marker='.', alpha=1, label='Pruned {}b'.format(nbits))

    #Plot "base"/unpruned model points
    for model, base_metric in zip(model_set,base_metrics_set):
        # base_metric = [[num_params],[base_metric]]
        nbits = model.weight_precision if hasattr(model, 'weight_precision') else 32
        rel_perf_ax.plot((base_metric[0] * nbits), 1/(base_metric[1]/base_metrics_set[0][1]), linestyle='solid', marker="X", alpha=1, label='Unpruned {}b'.format(nbits))

    rel_perf_ax.set_ylabel("1/{}/FP{}".format(metric_text,metric_text))
    rel_perf_ax.set_xlabel("Bit Params (Params * bits)")
    rel_perf_ax.grid(color='lightgray', linestyle='-', linewidth=1, alpha=0.3)
    rel_perf_ax.legend(loc='best')
    rel_perf_plt.savefig(path.join(outputDir, filename))
    rel_perf_plt.show()
    plt.close(rel_perf_plt)


def plot_total_loss(model_set, model_totalloss_set, model_estop_set, outputDir='..'):
    # Total loss over fine tuning
    now = datetime.now()
    time = now.strftime("%d-%m-%Y_%H-%M-%S")
    for model, model_loss, model_estop in zip(model_set, model_totalloss_set, model_estop_set):
        tloss_plt = plt.figure()
        tloss_ax = tloss_plt.add_subplot()
        nbits = model.weight_precision if hasattr(model, 'weight_precision') else 32
        filename = 'total_loss_{}b_{}.png'.format(nbits,time)
        tloss_ax.plot(range(1, len(model_loss[0]) + 1), model_loss[0], label='Training Loss')
        tloss_ax.plot(range(1, len(model_loss[1]) + 1), model_loss[1], label='Validation Loss')
        # plot each stopping point
        for stop in model_estop:
            tloss_ax.axvline(stop, linestyle='--', color='r', alpha=0.3)
        tloss_ax.set_xlabel('epochs')
        tloss_ax.set_ylabel('loss')
        tloss_ax.grid(True)
        tloss_ax.legend(loc='best')
        tloss_ax.set_title('Total Loss Across pruning & fine tuning {}b model'.format(nbits))
        tloss_plt.tight_layout()
        tloss_plt.savefig(path.join(outputDir,filename))
        tloss_plt.show()
        plt.close(tloss_plt)

def plot_total_eff(model_set, model_eff_set, model_estop_set, outputDir='..'):
    # Total loss over fine tuning
    now = datetime.now()
    time = now.strftime("%d-%m-%Y_%H-%M-%S")
    for model, model_eff_iter, model_estop in zip(model_set, model_eff_set, model_estop_set):
        tloss_plt = plt.figure()
        tloss_ax = tloss_plt.add_subplot()
        nbits = model.weight_precision if hasattr(model, 'weight_precision') else 32
        filename = 'total_eff_{}b_{}.png'.format(nbits,time)
        tloss_ax.plot(range(1, len(model_eff_iter) + 1), [z['net_efficiency'] for z in model_eff_iter], label='Net Efficiency',
                     color='green')

        # plot each stopping point
        for stop in model_estop:
            tloss_ax.axvline(stop, linestyle='--', color='r', alpha=0.3)
        tloss_ax.set_xlabel('epochs')
        tloss_ax.set_ylabel('Net Efficiency')
        tloss_ax.grid(True)
        tloss_ax.legend(loc='best')
        tloss_ax.set_title('Total Net. Eff. Across pruning & fine tuning {}b model'.format(nbits))
        tloss_plt.tight_layout()
        tloss_plt.savefig(path.join(outputDir,filename))
        tloss_plt.show()
        plt.close(tloss_plt)