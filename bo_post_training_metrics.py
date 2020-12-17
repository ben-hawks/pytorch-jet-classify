"""
pytorch_example.py - A demonstration of how to calculate neural efficiency metrics in pytorch

Step 1a: Install Brevitas (not on PyPi)
pip install git+https://github.com/Xilinx/brevitas.git

***Note that this is installing from the master branch and Brevitas is under active development, so some model files
generated with previous versions might not load or work properly. If this is the case, you can comment out the quantized
versions and run only the float model, which is the first in the list using models.three_layer_model_masked

Step 1b: Download Jet Tagging Test dataset (in this directory)
source ./jet_data_download.sh

Step 2: Install the requirements
pip install -r requirements.txt

Step 3: Run the example
python pytorch_example.py

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import jet_dataset
import models
import matplotlib.pyplot as plt
import ast
import re
from optparse import OptionParser
from tools.aiq import calc_AiQ
from tools.param_count import calc_BOPS, countNonZeroWeights
from tools.parse_yaml_config import parse_config

def gen_bo_model_dict(dir, bits=32):
    # Modified to load a set of BO models (with varying layer sizes)
    # Provide an instance of the model you're loading (to calculate # params)
    # and the path to folder containing model files, returns a dict with the format {pruned params:path to model}
    # and the total count of params in that model.
    model_dict = {}
    model_sizes = []
    first = True
    total_param = 0
    if os.path.isdir(dir):
        print("Directory found! Loading dir: " + dir)
        dir_list = os.listdir(dir)
        dir_list.sort()
        for file in dir_list:
            try:
                sizestr = re.search('(_\d\d?-\d\d?-\d\d?_)',file).group().strip('_').replace('-',', ') #Get the model side from the filename, just saves a bunch of headache
                dims = [int(m) for m in sizestr.split(',')]
                prune_masks = {
                    "fc1": torch.ones(dims[0], 16),
                    "fc2": torch.ones(dims[1], dims[0]),
                    "fc3": torch.ones(dims[2], dims[1]),
                    "fc4": torch.ones(5, dims[2])}
                bomodel = models.three_layer_model_bv_tunable(prune_masks,size,bits) #Shouldnt have to worry about correct precision for simple param count (for now)
                bomodel.load_state_dict(torch.load(os.path.join(dir, file), map_location=device))
                count, total_param, _, _ = countNonZeroWeights(bomodel)
                bops = calc_BOPS(bomodel)
                model_dict.update({int(bops): file})
            except Exception as e:
                print("Warning! Failed to load file " + file)
                print(e)
        return model_dict, total_param
    else:
        raise RuntimeError("Error! Unable to find directory " + dir)


""" Load the model and test dataset """
parser = OptionParser()
parser.add_option('-i','--input',action='store',type='string',dest='model_files',default='model_files/full_dataset/',
                  help='Directory containing sub directorys of model files (32b, 12b, 8b, 6b, 4b)')
parser.add_option('-o', '--output', action='store', type='string', dest='outputDir', default='train_simple/',
                  help='output directory')
parser.add_option('-t', '--test', action='store', type='string', dest='test', default='train_data/test',
                  help='Location of test data set')
parser.add_option('-c','--config', action='store',type='string',dest='config', default='/opt/repo/pytorch-jet-classify/configs/train_config_threelayer.yml', help='tree name')
parser.add_option('-n','--name', action='store',type='string',dest='name', default='aIQ_results.json', help='JSON Output Filename')
parser.add_option('-a', '--no_bn_affine', action='store_false', dest='bn_affine', default=True,
                  help='disable BN Affine Parameters')
parser.add_option('-s', '--no_bn_stats', action='store_false', dest='bn_stats', default=True,
                  help='disable BN running statistics')
parser.add_option('-b', '--no_batnorm', action='store_false', dest='batnorm', default=True,
                  help='disable BatchNormalization (BN) Layers ')
parser.add_option('-r', '--no_l1reg', action='store_false', dest='l1reg', default=True,
                  help='disable L1 Regularization totally ')

(options,args) = parser.parse_args()

if not os.path.exists(options.outputDir):  # create given output directory if it doesnt exist
    os.makedirs(options.outputDir)

# Load the model config file
yamlConfig = parse_config(options.config)

#Mask sets, required to load model class but not 'used' here really
prune_mask_set = {  # Float Model
        "fc1": torch.ones(64, 16),
        "fc2": torch.ones(32, 64),
        "fc3": torch.ones(32, 32),
    	"fc4": torch.ones(5, 32)}

# Setup cuda
use_cuda = False  # torch.cuda.is_available() #Don't really need CUDA here at the moment, so just set false
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Load datasets
#yrdy_dataset = jet_dataset.ParticleJetDataset('train_data/train/', yamlConfig)
test_dataset = jet_dataset.ParticleJetDataset(options.test, yamlConfig)

#train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=10000,
#                                         shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=25000,
                                          shuffle=False, num_workers=0)


dir = "model_files/"
dir = options.model_files


try:
    if options.batnorm:
        loadmodel = models.three_layer_model_batnorm_masked(prune_mask_set, bn_affine=options.bn_affine,
                                                        bn_stats=options.bn_stats)
    else:
        loadmodel = models.three_layer_model_masked(prune_mask_set)  # 32b

    float_model_set, model_max_params = gen_bo_model_dict(os.path.join(dir, '32b'),32)
except Exception as e:
    print(e)
    float_model_set, model_max_params = {},0


try:
    quant_model_set_4b, quant_4b_max_params = gen_bo_model_dict(os.path.join(dir, '4b'),4)
except:
    quant_model_set_4b, quant_4b_max_params = {},0


try:
    quant_model_set_6b, quant_6b_max_params = gen_bo_model_dict(os.path.join(dir, '6b'),6)
except:
    quant_model_set_6b, quant_6b_max_params = {},0


try:
    quant_model_set_12b, quant_12b_max_params = gen_bo_model_dict(os.path.join(dir, '12b'),12)
except:
    quant_model_set_12b, quant_12b_max_params = {},0

#Run through each model set, calculating AiQ for each model in the set
float_AiQ = {}
for model_bops, model_file in sorted(float_model_set.items()):
    sizestr = re.search('(_\d\d?-\d\d?-\d\d?_)',model_file).group().strip('_').replace('-',', ') #Get the model side from the filename, just saves a bunch of headache
    size = ast.literal_eval(sizestr)
    dims = size
    prune_masks = {
        "fc1": torch.ones(dims[0], 16),
        "fc2": torch.ones(dims[1], dims[0]),
        "fc3": torch.ones(dims[2], dims[1]),
        "fc4": torch.ones(5, dims[2])}
    print('Calculating AiQ for BO 32b, ' + str(model_bops) + ' BOPS, size ' + str(size))
    results = calc_AiQ(models.three_layer_model_bv_tunable(prune_masks,size,32), test_loader, loadfile=os.path.join(dir, '32b', model_file), batnorm = options.batnorm, device='cpu')
    results.update({'dims': dims})
    float_AiQ.update({model_bops: results})

quant_12b_AiQ = {}
for model_bops, model_file in sorted(quant_model_set_12b.items()):
    sizestr = re.search('(_\d\d?-\d\d?-\d\d?_)',model_file).group().strip('_').replace('-',', ') #Get the model side from the filename, just saves a bunch of headache
    size = ast.literal_eval(sizestr)
    dims = size
    prune_masks = {
        "fc1": torch.ones(dims[0], 16),
        "fc2": torch.ones(dims[1], dims[0]),
        "fc3": torch.ones(dims[2], dims[1]),
        "fc4": torch.ones(5, dims[2])}
    print('Calculating AiQ for BO 12b, ' + str(model_bops) + ' BOPS, size ' + str(size))
    results = calc_AiQ(models.three_layer_model_bv_tunable(prune_masks,size,12), test_loader, loadfile=os.path.join(dir, '12b', model_file), batnorm = options.batnorm, device='cpu')
    results.update({'dims': dims})
    quant_12b_AiQ.update({model_bops: results})


quant_4b_AiQ = {}
for model_bops, model_file in sorted(quant_model_set_4b.items()):
    sizestr = re.search('(_\d\d?-\d\d?-\d\d?_)',model_file).group().strip('_').replace('-',', ') #Get the model side from the filename, just saves a bunch of headache
    size = ast.literal_eval(sizestr)
    dims = size
    prune_masks = {
        "fc1": torch.ones(dims[0], 16),
        "fc2": torch.ones(dims[1], dims[0]),
        "fc3": torch.ones(dims[2], dims[1]),
        "fc4": torch.ones(5, dims[2])}
    print('Calculating AiQ for BO 4b, ' + str(model_bops) + ' BOPS, size ' + str(size))
    results = calc_AiQ(models.three_layer_model_bv_tunable(prune_masks,size,4), test_loader, loadfile=os.path.join(dir, '4b', model_file), batnorm = options.batnorm, device='cpu')
    results.update({'dims': dims})
    quant_4b_AiQ.update({model_bops: results})

quant_6b_AiQ = {}
for model_bops, model_file in sorted(quant_model_set_6b.items()):
    sizestr = re.search('(_\d\d?-\d\d?-\d\d?_)',model_file).group().strip('_').replace('-',', ') #Get the model side from the filename, just saves a bunch of headache
    size = ast.literal_eval(sizestr)
    dims = size
    prune_masks = {
        "fc1": torch.ones(dims[0], 16),
        "fc2": torch.ones(dims[1], dims[0]),
        "fc3": torch.ones(dims[2], dims[1]),
        "fc4": torch.ones(5, dims[2])}
    print('Calculating AiQ for BO 6b, ' + str(model_bops) + ' BOPS, size ' + str(size))
    results = calc_AiQ(models.three_layer_model_bv_tunable(prune_masks,size,6), test_loader, loadfile=os.path.join(dir, '6b', model_file), batnorm = options.batnorm, device='cpu')
    results.update({'dims': dims})
    quant_6b_AiQ.update({model_bops: results})

import json
dump_dict={ '32b':float_AiQ,
            '12b':quant_12b_AiQ,
            '6b':quant_6b_AiQ,
            '4b':quant_4b_AiQ
}

with open(os.path.join(options.outputDir, options.name+".json"), 'w') as fp:
    json.dump(dump_dict, fp)

best = {'4b': [44, 32, 32],
        '6b': [54, 32, 32],
        '12b': [64, 32, 19],
        '32b': [64, 28, 27]}

# AUCROC Plot
precisions = ['32b','12b','6b','4b']  #,'8b']  # What precisions to plot
colors = ['blue', 'green', 'red', 'orange', 'purple', 'pink']  # What colors to use for plots
for precision, color in zip(precisions, colors):
    eff_plot = plt.figure()
    eff_ax = eff_plot.add_subplot()
    eff_ax.grid(True)
    eff_ax.set_xlabel('BOPs')
    eff_ax.set_ylabel('AUC')
    eff_ax.set_xscale("symlog", linthresh=1e6)
    eff_ax.set_xlim(1e4, 1e7)
    eff_ax.set_ylim(0.45, 1)
    eff_ax.set_title("AUC vs BOPS - BO ({} HLS4ML Jet Tagging Model)".format(precision))
    eff_ax.scatter([int(key) for key in dump_dict[precision]], [z['auc_roc'] for z in dump_dict[precision].values()],
                   label='{}-bit'.format(precision.rstrip('b')), color=color, alpha=0.5)  # , marker='.',markersize=10,
    for txt, x, y in zip([str(z['dims']) for z in dump_dict[precision].values()], [int(key) for key in dump_dict[precision]],
                         [z['auc_roc'] for z in dump_dict[precision].values()]):
        if txt == str(best[precision]):
            eff_ax.annotate(txt, (x, y), color='black', label='_nolegend_')
            eff_ax.scatter(x, y, marker="*", s=75, label='Best {} ({})'.format(precision, txt), color=color,
                           edgecolor='black')
    eff_ax.legend(loc='best', title="Bayesian Optimization", framealpha=0.5)
    eff_plot.savefig('AUCROC_BO_{}.png'.format(precision))
    # eff_plot.savefig('AUCROC_FT_rand{}.pdf'.format(rand))
    eff_plot.show()

# Accuracy plot
for precision, color in zip(precisions, colors):
    eff_plot = plt.figure()
    eff_ax = eff_plot.add_subplot()
    eff_ax.grid(True)
    eff_ax.set_xlabel('BOPs')
    eff_ax.set_ylabel('Accuracy')
    eff_ax.set_xscale("symlog", linthresh=1e6)
    eff_ax.set_xlim(1e4, 1e7)
    eff_ax.set_ylim(0.6, 0.8)
    eff_ax.set_title("Accuracy vs BOPS - BO ({} HLS4ML Jet Tagging Model)".format(precision))
    eff_ax.scatter([int(key) for key in dump_dict[precision]], [z['accuracy'] for z in dump_dict[precision].values()],
                   label='{}-bit'.format(precision.rstrip('b')), color=color, alpha=0.5)  # , marker='.',markersize=10,
    for txt, x, y in zip([str(z['dims']) for z in dump_dict[precision].values()], [int(key) for key in dump_dict[precision]],
                         [z['accuracy'] for z in dump_dict[precision].values()]):
        if txt == str(best[precision]):
            eff_ax.annotate(txt, (x, y), color='black', label='_nolegend_')
            eff_ax.scatter(x, y, marker="*", s=75, label='Best {} ({})'.format(precision, txt), color=color,
                           edgecolor='black')
    eff_ax.legend(loc='best', title="Bayesian Optimization", framealpha=0.5)
    eff_plot.savefig('ACC_BO_{}.png'.format(precision))
    # eff_plot.savefig('ACC_FT_rand{}.pdf'.format(rand))
    eff_plot.show()

# Efficiency plot
for precision, color in zip(precisions, colors):
    eff_plot = plt.figure()
    eff_ax = eff_plot.add_subplot()
    eff_ax.grid(True)
    eff_ax.set_xlabel('BOPs')
    eff_ax.set_ylabel('Efficiency')
    eff_ax.set_xscale("symlog", linthresh=1e6)
    eff_ax.set_xlim(1e4, 1e7)
    eff_ax.set_ylim(0, 0.6)
    eff_ax.set_title("Efficiency vs BOPS BO ({} HLS4ML Jet Tagging Model)".format(precision))
    eff_ax.scatter([int(key) for key in dump_dict[precision]], [z['net_efficiency'] for z in dump_dict[precision].values()],
                   label='{}-bit'.format(precision.rstrip('b')), color=color, alpha=0.5)  # , marker='.',markersize=10,
    for txt, x, y in zip([str(z['dims']) for z in dump_dict[precision].values()], [int(key) for key in dump_dict[precision]],
                         [z['net_efficiency'] for z in dump_dict[precision].values()]):
        if txt == str(best[precision]):
            eff_ax.annotate(txt, (x, y), color='black', label='_nolegend_')
            eff_ax.scatter(x, y, marker="*", s=75, label='Best {} ({})'.format(precision, txt), color=color,
                           edgecolor='black')
    eff_ax.legend(loc='best', title="Bayesian Optimization", framealpha=0.5)
    eff_plot.savefig('Eff_BO_{}.png'.format(precision))
    # eff_plot.savefig('ACC_FT_rand{}.pdf'.format(rand))
    eff_plot.show()

