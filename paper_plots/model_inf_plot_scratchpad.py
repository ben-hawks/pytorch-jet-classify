# Import misc packages
import math
import json
import os
import os.path as path
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, auc
import matplotlib.pyplot as plt
from matplotlib import lines
from optparse import OptionParser
import pandas as pd

# Import torch stuff
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.prune as prune

# Import our own code
import models
import jet_dataset
from training.early_stopping import EarlyStopping
from training.train_funcs import train, val  # ,test_pruned as test
from tools.aiq import calc_AiQ
from training.training_plots import plot_total_loss, plot_total_eff, plot_metric_vs_bitparam, plot_kernels
from tools.param_count import countNonZeroWeights
from tools.parse_yaml_config import parse_config

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import mplhep as hep
hep.set_style(hep.style.ROOT)

# Quick and dirty script to generate a few misc plots, specifically those requiring an inference
# on the test set to be run, and another because im too lazy to make it a seperate script


def loss_plot(precision, model_loss, model_estop,label="", outputDir='..'):
#{nbits:[model_loss,model_eff,model_estop]})
# Total loss over fine tuning
    tloss_plt = plt.figure()
    tloss_ax = tloss_plt.add_subplot()
    nbits = precision
    filename = 'total_loss_{}b_{}.pdf'.format(nbits, label)
    tloss_ax.plot(range(1, len(model_loss[0]) + 1), model_loss[0], label='Training Loss')
    tloss_ax.plot(range(1, len(model_loss[1]) + 1), model_loss[1], label='Validation Loss')
    # plot each stopping point
    for i, stop  in enumerate(model_estop):
        if i > 0:
            tloss_ax.axvline(stop, linestyle='--', color='r', alpha=0.3)
        else:
            tloss_ax.axvline(stop, linestyle='--', color='r', alpha=0.3, label='Iteration Stop')
    tloss_ax.set_xlabel('Epochs')
    tloss_ax.set_ylabel('Loss')
    tloss_ax.grid(True)
    tloss_ax.legend(loc='best', title=label)
    tloss_plt.tight_layout()
    tloss_plt.savefig(path.join(outputDir, filename))
    tloss_plt.show()
    plt.close(tloss_plt)

def test(model, model2, test_loader, plot=True, nbits=32, outputDir='..', device='cpu', test_dataset_labels=[]):
    #device = torch.device('cpu') #required if doing a untrained init check
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    predlist2 = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist2 = torch.zeros(0, dtype=torch.long, device='cpu')
    model.to(device)
    model.mask_to_device(device)
    model2.to(device)
    model2.mask_to_device(device)
    with torch.no_grad():  # Evaulate pruned model performance
        for i, data in enumerate(test_loader):

            model.eval()

            local_batch, local_labels = data
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)


            #model 1
            outputs = model(local_batch.float())
            _, preds = torch.max(outputs, 1)
            predlist = torch.cat([predlist, preds.view(-1).cpu()])
            lbllist = torch.cat([lbllist, torch.max(local_labels, 1)[1].view(-1).cpu()])


        for i, data in enumerate(test_loader):
            model2.eval()

            local_batch2, local_labels2 = data
            local_batch2, local_labels2 = local_batch2.to(device), local_labels2.to(device)

            #model 2
            outputs2 = model2(local_batch2.float())
            _, preds2 = torch.max(outputs2, 1)
            predlist2 = torch.cat([predlist2, preds2.view(-1).cpu()])
            lbllist2 = torch.cat([lbllist2, torch.max(local_labels2, 1)[1].view(-1).cpu()])

        outputs = outputs.cpu()
        outputs2 = outputs2.cpu()
        local_labels = local_labels.cpu()
        local_labels2 = local_labels2.cpu()

        if plot:
            predict_test = outputs.numpy()
            df = pd.DataFrame()
            fpr = {}
            tpr = {}
            auc1 = {}

            predict_test2 = outputs2.numpy()
            df2 = pd.DataFrame()
            fpr2 = {}
            tpr2 = {}
            auc2 = {}

            #Time for filenames
            now = datetime.now()
            time = now.strftime("%d-%m-%Y_%H-%M-%S")

            # AUC/Signal Efficiency
            filename = 'ROC_Unpruned_FT_32_6.png'
            filename2 = 'ROC_Unpruned_FT_32_6.pdf'

            sig_eff_plt = plt.figure()
            sig_eff_ax = sig_eff_plt.add_subplot()
            colors = ['blue', 'green', 'red', 'orange', 'purple', 'pink']
            for (i, label), color in zip(enumerate(test_dataset_labels),colors):

                df[label] = local_labels[:, i]
                df[label + '_pred'] = predict_test[:, i]
                fpr[label], tpr[label], threshold = roc_curve(np.nan_to_num(df[label]), np.nan_to_num(df[label + '_pred']))
                auc1[label] = auc(np.nan_to_num(fpr[label]), np.nan_to_num(tpr[label]))
                sig_eff_ax.plot(np.nan_to_num(tpr[label]), np.nan_to_num(fpr[label]), linestyle='solid', color=color,
                         label='32b %s tagger, AUC = %.1f%%' % (label.replace('j_', ''), np.nan_to_num(auc1[label]) * 100.))

                df2[label] = local_labels2[:, i]
                df2[label + '_pred'] = predict_test2[:, i]
                fpr2[label], tpr2[label], threshold = roc_curve(np.nan_to_num(df2[label]), np.nan_to_num(df2[label + '_pred']))
                auc2[label] = auc(np.nan_to_num(fpr2[label]), np.nan_to_num(tpr2[label]))
                sig_eff_ax.plot(np.nan_to_num(tpr2[label]), np.nan_to_num(fpr2[label]), linestyle='dotted', color=color,
                         label='6b %s tagger, AUC = %.1f%%' % (label.replace('j_', ''), np.nan_to_num(auc2[label]) * 100.))

            sig_eff_ax.set_yscale('log')
            sig_eff_ax.set_xlabel("Signal Efficiency")
            sig_eff_ax.set_ylabel("Background Efficiency")
            sig_eff_ax.set_ylim(0.001, 1)
            sig_eff_ax.grid(True)
            sig_eff_ax.legend(loc='upper left',fontsize=18)
            sig_eff_plt.savefig(path.join(outputDir, filename))
            sig_eff_plt.savefig(path.join(outputDir, filename2))
            sig_eff_plt.show()
            plt.close(sig_eff_plt)

    return


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-o','--output'   ,action='store',type='string',dest='outputDir'   ,default='train_simple/', help='output directory')
    parser.add_option('-t','--test'   ,action='store',type='string',dest='test'   ,default='train_data/test', help='Location of test data set')
    parser.add_option('-c','--config'   ,action='store',type='string',dest='config'   ,default='configs/train_config_threelayer.yml', help='tree name')
    (options,args) = parser.parse_args()
    yamlConfig = parse_config(options.config)

    # create given output directory if it doesnt exist
    if not path.exists(options.outputDir):
        os.makedirs(options.outputDir, exist_ok=True)

    loss_FT_file = "../model_losses_6_FT.json"
    loss_LT_file = "../model_losses_6_LT.json"

    with open(loss_FT_file, "r") as read_file:
        loss_FT = json.load(read_file)
    with open(loss_LT_file, "r") as read_file:
        loss_LT = json.load(read_file)

    loss_FT_trainloss = loss_FT['6'][0]
    loss_LT_trainloss = loss_LT['6'][0]

    loss_FT_estops = loss_FT['6'][2]
    loss_LT_estops = loss_LT['6'][2]

    loss_plot(6,loss_FT_trainloss,loss_FT_estops, label="Fine Tuning")
    loss_plot(6,loss_LT_trainloss,loss_LT_estops, label="Lottery Ticket")

    # Setup cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Using Device: {}".format(device))
    if use_cuda:
        print("cuda:0 device type: {}".format(torch.cuda.get_device_name(0)))
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.fastest = True

    # Set Batch size and split value
    batch_size = 1024

    standard_mask = {
            "fc1": torch.ones(64, 16),
            "fc2": torch.ones(32, 64),
            "fc3": torch.ones(32, 32),
            "fc4": torch.ones(5, 32)}

    prune_mask_set = [standard_mask for m in range(0, 2)]

    print("Made {} Masks!".format(len(prune_mask_set)))

    # First model should be the "Base" model that all other accuracies are compared to!

    test_dataset = jet_dataset.ParticleJetDataset(options.test, yamlConfig)
    test_size = len(test_dataset)

    print("test dataset size: " + str(len(test_dataset)))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size,
                                              shuffle=False, num_workers=10, pin_memory=True)

    m6_file = "../6b_unpruned_FT0.pth"
    m32_file = "../32b_unpruned_FT0.pth"

    m32 = models.three_layer_model_batnorm_masked(prune_mask_set[0], bn_affine=True, bn_stats=True)
    m6 = models.three_layer_model_bv_batnorm_masked(prune_mask_set[1], 6, bn_affine=True, bn_stats=True)

    m32.load_state_dict(torch.load(os.path.join(m32_file), map_location=device))
    m6.load_state_dict(torch.load(os.path.join(m6_file), map_location=device))

    test(m32, m6, test_loader,
         outputDir=options.outputDir,
         device=device, test_dataset_labels=test_dataset.labels_list)