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

import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tools import TensorEfficiency
import torch
import jet_dataset
import models
import yaml
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, mean_squared_error
import numpy as np
import brevitas.nn as qnn
import matplotlib.pyplot as plt
import ast
import re
import math
from optparse import OptionParser


#Class to register as forward hook to collect outputs of intermediate layers
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

def parse_config(config_file) :
    print("Loading configuration from", config_file)
    config = open(config_file, 'r')
    return yaml.load(config, Loader=yaml.FullLoader)


def plot_metric_vs_bitparam(model_set,metric_results_set,bit_params_set,base_metrics_set,metric_text):
    # NOTE: Assumes that the first object in the base metrics set is the true base of comparison
    # now = datetime.now()
    #time = now.strftime("%d-%m-%Y_%H-%M-%S")

    #filename = '{}_vs_bitparams'.format(metric_text) + str(time) + '.png'

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
    rel_perf_plt.savefig("rel_{}_BOPS.png".format(metric_text))
    rel_perf_plt.show()

def ae_MSE(model, model_file, test_loaders):
    errors_list = []
    model.load_state_dict(torch.load(os.path.join(model_file), map_location=device))
    for test_loader in test_loaders: #Eventually evaluate the ROC if we're using the Dev set (vs Eval set), just returm MSE for now
        with torch.no_grad():  # Evaulate pruned model performance
            for i, data in enumerate(test_loader):
                model.eval()
                local_batch, local_labels = data
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                outputs = model(local_batch.float())
            outputs = outputs.cpu()
            local_labels = local_labels.cpu().numpy()
            predict_test = outputs.numpy()
            #errors = np.mean(np.square(data - predict_test), axis=1) #MSE of input vs output
            errors = mean_squared_error(local_labels, predict_test)
            errors_list.append(errors)
    print("MSE: {}".format(np.mean(errors_list)))
    return np.mean(errors_list)

def gen_model_dict(bopmodel, dir):
    # Provide an instance of the model you're loading (to calculate # params)
    # and the path to folder containing model files, returns a dict with the format {pruned params:path to model}
    # and the total count of params in that model. Excepts if a model with a different total param count is found
    model_dict = {}
    first = True
    total_param = 0
    if os.path.isdir(dir):
        print("Directory found! Loading dir: " + dir)
        dir_list = os.listdir(dir)
        dir_list.sort()
        #print(dir_list)
        for file in dir_list:
            try:
                bopmodel.load_state_dict(torch.load(os.path.join(dir, file), map_location=device))
                count, total_cnt, _, _ = countNonZeroWeights(bopmodel)
                bops = calc_BOPS(bopmodel)
                if first: #Assume first alphabetical is the first model, for the sake of checking all pth are same model
                    total_param = total_cnt
                    first = False
                else:
                    if total_cnt != total_param:
                        raise RuntimeError("Error! Model mismatch while creating model dict! Expected {} total params, found {}".format(total_param, total_cnt))

                model_dict.update({int(bops): file})
            except Exception as e:
                print("Warning! Failed to load file " + file)
                print(e)
        return model_dict, total_param
    else:
        raise RuntimeError("Error! Unable to find directory " + dir)

def gen_bo_model_dict(dir):
    ### WARNING ONLY USED A TRUSTED RESULTS FILE, THIS USES EVAL WHICH IS KIND OF BAD AND VERY UNSAFE OTHERWISE ###
    # Modified to load a set of BO models (with varying layer sizes)
    # Provide an instance of the model you're loading (to calculate # params)
    # and the path to folder containing model files, returns a dict with the format {pruned params:path to model}
    # and the total count of params in that model. Excepts if a model with a different total param count is found
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
                size = ast.literal_eval(sizestr)
                dims = size
                prune_masks = {
                "fc1": torch.ones(dims[0], 16),
                "fc2": torch.ones(dims[1], dims[0]),
                "fc3": torch.ones(dims[2], dims[1])}
                bomodel = models.three_layer_model_bv_tunable(prune_masks,size) #Shouldnt have to worry about correct precision for simple param count (for now)
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

def countNonZeroWeights(model):
    nonzero = total = 0
    layer_count_alive = {}
    layer_count_total = {}
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        layer_count_alive.update({name: nz_count})
        layer_count_total.update({name: total_params})
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    return nonzero, total, layer_count_alive, layer_count_total

def calc_BOPS(model, input_data_precision=32):
    last_bit_width = input_data_precision
    alive, total,l_alive,l_total = countNonZeroWeights(model)
    b_w = model.weight_precision if hasattr(model, 'weight_precision') else 32
    total_BOPS = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, qnn.QuantLinear):
            b_a = last_bit_width
            #b_w = module.quant_weight_bit_width #Dont think this is a property I can access sadly, going with precision as given set in model
            n = module.in_features
            m = module.out_features
            total = l_total[name+'.weight'] + l_total[name+'.bias']
            alive = l_alive[name + '.weight'] + l_alive[name + '.bias']
            p = 1 - ((total - alive) / total)  # fraction of layer remaining
            #assuming b_a is the output bitwidth of the last layer
            #module_BOPS = m*n*p*(b_a*b_w + b_a + b_w + math.log2(n))
            module_BOPS = m * n * (p * b_a * b_w + b_a + b_w + math.log2(n))
            print("{} BOPS: {} = {}*{}({}*{}*{} + {} + {} + {})".format(name,module_BOPS,m,n,p,b_a,b_w,b_a,b_w,math.log2(n)))
            last_bit_width = b_w
            total_BOPS += module_BOPS
    print("Total BOPS: {}".format(total_BOPS))
    return total_BOPS

def calc_AiQ(model,model_file):
    """ Calculate efficiency of network using TensorEfficiency """
    model.load_state_dict(torch.load(os.path.join(model_file), map_location=device))
    # Time the execution
    start_time = time.time()

    # Set up the data
    ensemble = {}
    accuracy = 0
    accuracy_list = []
    roc_list = []
    sel_bkg_reject_list = []
    # Initialize arrays for storing microstates
    microstates = {name: np.ndarray([]) for name, module in model.named_modules() if
                   ((isinstance(module, torch.nn.Linear) or isinstance(module, qnn.QuantLinear)) and name == 'fc4') \
                   or (isinstance(module, torch.nn.BatchNorm1d))}
    microstates_count = {name: 0 for name, module in model.named_modules() if
                         ((isinstance(module, torch.nn.Linear) or isinstance(module,qnn.QuantLinear)) and name == 'fc4') \
                         or (isinstance(module, torch.nn.BatchNorm1d))}

    activation_outputs = SaveOutput()  # Our forward hook class, stores the outputs of each layer it's registered to

    # register a forward hook to get and store the activation at each Linear layer while running
    layer_list = []
    for name, module in model.named_modules():
        if ((isinstance(module, torch.nn.Linear) or isinstance(module, qnn.QuantLinear)) and name == 'fc4') \
          or (isinstance(module, torch.nn.BatchNorm1d)):  # Record @ BN output except last layer (since last has no BN)
            module.register_forward_hook(activation_outputs)
            layer_list.append(name)  # Probably a better way to do this, but it works,

    # Process data using torch dataloader, in this case we
    for i, data in enumerate(test_loader, 0):
        activation_outputs.clear()
        local_batch, local_labels = data

        # Run through our test batch and get inference results
        with torch.no_grad():
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch.float())

        # Calculate accuracy (top-1 averaged over each of n=5 classes)
        outputs.cpu()
        predlist = torch.zeros(0, dtype=torch.long, device='cpu')
        lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
        _, preds = torch.max(outputs, 1)
        predlist = torch.cat([predlist, preds.view(-1).cpu()])
        lbllist = torch.cat([lbllist, torch.max(local_labels, 1)[1].view(-1).cpu()])
        accuracy_list.append(np.average((accuracy_score(lbllist.numpy(), predlist.numpy()))))
        roc_list.append(roc_auc_score(np.nan_to_num(local_labels.numpy()), np.nan_to_num(outputs.numpy())))

        #Calculate background eff @ signal eff of 50%
        df = pd.DataFrame()
        fpr = {}
        tpr = {}
        auc1 = {}
        bkg_reject = {}
        predict_test = outputs.numpy()
        # X = TPR, Y = FPR
        for i, label in enumerate(test_dataset.labels_list):
            df[label] = local_labels[:, i]
            df[label + '_pred'] = predict_test[:, i]
            fpr[label], tpr[label], threshold = roc_curve(np.nan_to_num(df[label]), np.nan_to_num(df[label + '_pred']))
            bkg_reject[label] = np.interp(0.5, np.nan_to_num(tpr[label]), (np.nan_to_num(fpr[label]))) # Get background rejection factor @ Sig Eff = 50%
            auc1[label] = auc(np.nan_to_num(fpr[label]), np.nan_to_num(tpr[label]))
        sel_bkg_reject_list.append(bkg_reject)

        #Calculate microstates for this run
        for name, x in zip(layer_list, activation_outputs.outputs):
            # print("---- AIQ Calc ----")
            # print("Act list: " + name + str(x))
            x = x.numpy()
            # Initialize the layer in the ensemble if it doesn't exist
            if name not in ensemble.keys():
                ensemble[name] = {}

            # Initialize an array for holding layer states if it has not already been initialized
            sort_count_freq = 5  # How often (iterations) we sort/count states
            if microstates[name].size == 1:
                microstates[name] = np.ndarray((sort_count_freq * np.prod(x.shape[0:-1]), x.shape[-1]), dtype=bool,
                                               order='F')

            # Store the layer states
            new_count = microstates_count[name] + np.prod(x.shape[0:-1])
            microstates[name][
            microstates_count[name]:microstates_count[name] + np.prod(x.shape[0:-1]), :] = np.reshape(x > 0,
                                                                                                      (-1, x.shape[-1])
                                                                                                      , order='F')
            # Only sort/count states every 5 iterations
            if new_count < microstates[name].shape[0]:
                microstates_count[name] = new_count
                continue
            else:
                microstates_count[name] = 0

            # TensorEfficiency.sort_microstates aggregates microstates by sorting
            sorted_states, index = TensorEfficiency.sort_microstates(microstates[name], True)

            # TensorEfficiency.accumulate_ensemble stores the the identity of each observed
            # microstate and the number of times that microstate occurred
            TensorEfficiency.accumulate_ensemble(ensemble[name], sorted_states, index)
        # If the current layer is the final layer, record the class prediction
        # if isinstance(module, torch.nn.Linear) or isinstance(module, qnn.QuantLinear):

    # Calculate efficiency and entropy of each layer
    layer_metrics = {}
    metrics = ['efficiency', 'entropy', 'max_entropy']
    for layer, states in ensemble.items():
        layer_metrics[layer] = {key: value for key, value in zip(metrics, TensorEfficiency.layer_efficiency(states))}
    sel_bkg_reject = {}
    accuracy = np.average(accuracy_list)
    auc_roc = np.average(roc_list)
    #print(sel_bkg_reject_list)
    for label in test_dataset.labels_list:
        sel_bkg_reject.update({label: np.average([batch[label] for batch in sel_bkg_reject_list])})
    metric = auc_roc #auc_roc or accuracy
    # Calculate network efficiency and aIQ, with beta=2
    net_efficiency = TensorEfficiency.network_efficiency([m['efficiency'] for m in layer_metrics.values()])
    aiq = TensorEfficiency.aIQ(net_efficiency, metric, 2)

    # Display information
    print('---------------------------------------')
    print('Network analysis using TensorEfficiency')
    print('---------------------------------------')
    for layer, metrics in layer_metrics.items():
        print('({}) Layer Efficiency: {}'.format(layer, metrics['efficiency']))
    print('Network Efficiency: {}'.format(net_efficiency))
    print('Accuracy: {}'.format(accuracy))
    print('AUC ROC Score: {}'.format(auc_roc))
    print('aIQ: {}'.format(aiq))
    print('Execution time: {}'.format(time.time() - start_time))
    print('')
    #Return AiQ along with our metrics
    return {'AiQ':aiq,
            'accuracy':accuracy,
            'auc_roc':auc_roc,
            'net_efficiency':net_efficiency,
            'sel_bkg_reject':sel_bkg_reject,
            'layer_metrics':layer_metrics}


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
prune_mask_set2 = [
    {  # 1/4 Quant Model
        "fc1": torch.ones(16, 16),
        "fc2": torch.ones(8, 16),
        "fc3": torch.ones(8, 8)},
    {  # 4x Quant Model
        "fc1": torch.ones(256, 16),
        "fc2": torch.ones(128, 256),
        "fc3": torch.ones(128, 128)}]
ae_prune_mask_set = {  
        "enc1": torch.ones(128, 640),
        "enc2": torch.ones(128, 128),
        "enc3": torch.ones(128, 128),
        "enc4": torch.ones(8, 128),
        "dec1": torch.ones(128, 8),
        "dec2": torch.ones(128, 128),
        "dec3": torch.ones(128, 128),
        "dec4": torch.ones(128, 128),
        "dout": torch.ones(640, 128),
    }
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
        # models.three_layer_model_bv_batnorm_masked(prune_mask_set,12, bn_affine=options.bn_affine, bn_stats=options.bn_stats)
    else:
        loadmodel = models.three_layer_model_masked(prune_mask_set),  # 32b
        # models.three_layer_model_bv_masked(prune_mask_set[1], 12)  # 12b

    float_model_set, model_max_params = gen_model_dict(loadmodel, os.path.join(dir, '32b'))
except:
    float_model_set, model_max_params = {},0


try:
    if options.batnorm:
        loadmodel = models.three_layer_model_bv_batnorm_masked(prune_mask_set,4, bn_affine=options.bn_affine, bn_stats=options.bn_stats)
    else:
        loadmodel = models.three_layer_model_bv_masked(prune_mask_set, 4)

    quant_model_set_4b, quant_4b_max_params = gen_model_dict(loadmodel, os.path.join(dir, '4b'))
except:
    quant_model_set_4b, quant_4b_max_params = {},0


try:
    if options.batnorm:
        loadmodel = models.three_layer_model_bv_batnorm_masked(prune_mask_set,6, bn_affine=options.bn_affine, bn_stats=options.bn_stats)
    else:
        loadmodel = models.three_layer_model_bv_masked(prune_mask_set, 6)

    quant_model_set_6b, quant_6b_max_params = gen_model_dict(loadmodel, os.path.join(dir, '6b'))
except:
    quant_model_set_6b, quant_6b_max_params = {},0


try:
    if options.batnorm:
        loadmodel = models.three_layer_model_bv_batnorm_masked(prune_mask_set,12, bn_affine=options.bn_affine, bn_stats=options.bn_stats)
    else:
        loadmodel = models.three_layer_model_bv_masked(prune_mask_set, 12)

    quant_model_set_12b, quant_12b_max_params = gen_model_dict(loadmodel, os.path.join(dir, '12b'))
except:
    quant_model_set_12b, quant_12b_max_params = {},0


try:
    if options.batnorm:
        loadmodel = models.three_layer_model_bv_batnorm_masked(prune_mask_set,8, bn_affine=options.bn_affine, bn_stats=options.bn_stats)
    else:
        loadmodel = models.three_layer_model_bv_masked(prune_mask_set, 8)

    quant_batnorm_model_set, batnorm_max_params = gen_model_dict(loadmodel, os.path.join(dir, '8b'))
except:
    quant_batnorm_model_set, batnorm_max_params = {},0

#Run through each model set, calculating AiQ for each model in the set
float_AiQ = {}
for model_bops, model_file in sorted(float_model_set.items()):
    if options.batnorm:
        loadmodel = models.three_layer_model_batnorm_masked(prune_mask_set, bn_affine=options.bn_affine,
                                                        bn_stats=options.bn_stats)
    else:
        loadmodel = models.three_layer_model_masked(prune_mask_set),  # 32b

    print('Calculating AiQ for 32b, ' + str(model_bops) + ' BOPS')
    float_AiQ.update({model_bops: calc_AiQ(loadmodel, os.path.join(dir, '32b', model_file))})

quant_4b_AiQ = {}
for model_bops, model_file in sorted(quant_model_set_4b.items()):
    if options.batnorm:
        loadmodel = models.three_layer_model_bv_batnorm_masked(prune_mask_set,4, bn_affine=options.bn_affine, bn_stats=options.bn_stats)
    else:
        loadmodel = models.three_layer_model_bv_masked(prune_mask_set, 4)
    print('Calculating AiQ for 4b, ' + str(model_bops) + ' BOPS')
    quant_4b_AiQ.update({model_bops: calc_AiQ(loadmodel, os.path.join(dir, '4b', model_file))})

quant_6b_AiQ = {}
for model_bops, model_file in sorted(quant_model_set_6b.items()):
    if options.batnorm:
        loadmodel = models.three_layer_model_bv_batnorm_masked(prune_mask_set,6, bn_affine=options.bn_affine, bn_stats=options.bn_stats)
    else:
        loadmodel = models.three_layer_model_bv_masked(prune_mask_set, 6)
    print('Calculating AiQ for 6b, ' + str(model_bops) + ' BOPS')
    quant_6b_AiQ.update({model_bops: calc_AiQ(loadmodel, os.path.join(dir, '6b',model_file))})

quant_12b_AiQ = {}
for model_bops, model_file in sorted(quant_model_set_12b.items()):
    if options.batnorm:
        loadmodel = models.three_layer_model_bv_batnorm_masked(prune_mask_set,12, bn_affine=options.bn_affine, bn_stats=options.bn_stats)
    else:
        loadmodel = models.three_layer_model_bv_masked(prune_mask_set, 12)
    print('Calculating AiQ for 12b, ' + str(model_bops) + ' BOPS')
    quant_12b_AiQ.update({model_bops: calc_AiQ(loadmodel, os.path.join(dir, '12b', model_file))})

quant_batnorm_AiQ = {}
for model_bops, model_file in sorted(quant_batnorm_model_set.items()):
    if options.batnorm:
        loadmodel = models.three_layer_model_bv_batnorm_masked(prune_mask_set,8, bn_affine=options.bn_affine, bn_stats=options.bn_stats)
    else:
        loadmodel = models.three_layer_model_bv_masked(prune_mask_set, 8)
    print('Calculating AiQ for 8b w/ BatchNorm, ' + str(model_bops) + ' BOPS')
    quant_batnorm_AiQ.update({model_bops: calc_AiQ(loadmodel, os.path.join(dir,'8b',model_file))})

import json
dump_dict={ '32b':float_AiQ,
            '12b':quant_12b_AiQ,
            '8b':quant_batnorm_AiQ,
            '6b':quant_6b_AiQ,
            '4b':quant_4b_AiQ
}
with open(os.path.join(options.outputDir, options.name+".json"), 'w') as fp:
    json.dump(dump_dict, fp)


aiq_plot = plt.figure()
aiq_ax = aiq_plot.add_subplot()
aiq_ax.set_title("aIQ vs BOPS (HLS4ML Jet Tagging Model w/ LTH)")
aiq_ax.grid(True)
aiq_ax.set_xscale("log")
aiq_ax.set_xlabel('Binary Operations (BOPS)')
aiq_ax.set_ylabel('AiQ (AUC ROC Score, B=2)')
#aiq_ax.plot([key for key in quant_AiQ],[z['AiQ'] for z in quant_AiQ.values()], label='8b Quant - No BatNorm')
aiq_ax.plot([key for key in float_AiQ],[z['AiQ'] for z in float_AiQ.values()], label='32b Float')
aiq_ax.plot([key for key in quant_12b_AiQ],[z['AiQ'] for z in quant_12b_AiQ.values()], label='12b Quantized')
aiq_ax.plot([key for key in quant_6b_AiQ],[z['AiQ'] for z in quant_6b_AiQ.values()], label='6b Quantized')
aiq_ax.plot([key for key in quant_4b_AiQ],[z['AiQ'] for z in quant_4b_AiQ.values()], label='4b Quantized')
#aiq_ax.plot([key for key in quant_batnorm_AiQ],[z['AiQ'] for z in quant_batnorm_AiQ.values()], label='8b Quantized')
aiq_ax.legend(loc='best')
aiq_plot.savefig('aIQ_BOPS.png')
aiq_plot.show()


aiq_plot = plt.figure()
aiq_ax = aiq_plot.add_subplot()
aiq_ax.set_title("AUC ROC vs BOPS (HLS4ML Jet Tagging Model w/ LTH)")
aiq_ax.grid(True)
aiq_ax.set_xscale("log")
aiq_ax.set_xlabel('Binary Operations (BOPS)')
aiq_ax.set_ylabel('AUC ROC Score')
#aiq_ax.plot([key for key in quant_AiQ],[z['auc_roc'] for z in quant_AiQ.values()], label='8b Quant - No BatNorm')
aiq_ax.plot([key for key in float_AiQ],[z['auc_roc'] for z in float_AiQ.values()], label='32b Float')
aiq_ax.plot([key  for key in quant_12b_AiQ],[z['auc_roc'] for z in quant_12b_AiQ.values()], label='12b Quantized')
aiq_ax.plot([key for key in quant_6b_AiQ],[z['auc_roc'] for z in quant_6b_AiQ.values()], label='6b Quantized')
aiq_ax.plot([key for key in quant_4b_AiQ],[z['auc_roc'] for z in quant_4b_AiQ.values()], label='4b Quantized')
#aiq_ax.plot([key for key in quant_batnorm_AiQ],[z['auc_roc'] for z in quant_batnorm_AiQ.values()], label='8b Quantized')
aiq_ax.legend(loc='best')
aiq_plot.savefig(os.path.join(options.outputDir,'auc_roc_BOPS.png'))
aiq_plot.show()


aiq_plot = plt.figure()
aiq_ax = aiq_plot.add_subplot()
aiq_ax.set_title("Accuracy vs BOPS (HLS4ML Jet Tagging Model w/ LTH)")
aiq_ax.grid(True)
aiq_ax.set_xlabel('Binary Operations (BOPS)')
aiq_ax.set_ylabel('Accuracy')
aiq_ax.set_xscale("log")
#aiq_ax.plot([key for key in quant_AiQ],[z['accuracy'] for z in quant_AiQ.values()], label='8b Quant - No BatNorm')
aiq_ax.plot([key for key in float_AiQ],[z['accuracy'] for z in float_AiQ.values()], label='32b Float')
aiq_ax.plot([key for key in quant_12b_AiQ],[z['accuracy'] for z in quant_12b_AiQ.values()], label='12b Quantized')
aiq_ax.plot([key for key in quant_6b_AiQ],[z['accuracy'] for z in quant_6b_AiQ.values()], label='6b Quantized')
aiq_ax.plot([key for key in quant_4b_AiQ],[z['accuracy'] for z in quant_4b_AiQ.values()], label='4b Quantized')
#aiq_ax.plot([key for key in quant_batnorm_AiQ],[z['accuracy'] for z in quant_batnorm_AiQ.values()], label='8b Quantized')
aiq_ax.legend(loc='best')
aiq_plot.savefig(os.path.join(options.outputDir,'acc_BOPS.png'))
aiq_plot.show()



eff_plot = plt.figure()
eff_ax = eff_plot.add_subplot()
eff_ax.set_title("Efficiency vs BOPS (HLS4ML Jet Tagging Model w/ LTH)")
eff_ax.grid(True)
eff_ax.set_xlabel('Binary Operations (BOPS)')
eff_ax.set_ylabel('Efficiency')
eff_ax.set_xscale("log")
#eff_ax.plot([key for key in quant_AiQ],[z['net_efficiency'] for z in quant_AiQ.values()], label='8b Quant - No BatNorm')
eff_ax.plot([key for key in float_AiQ],[z['net_efficiency'] for z in float_AiQ.values()], label='32b Float')
eff_ax.plot([key for key in quant_12b_AiQ],[z['net_efficiency'] for z in quant_12b_AiQ.values()], label='12b Quantized')
eff_ax.plot([key for key in quant_6b_AiQ],[z['net_efficiency'] for z in quant_6b_AiQ.values()], label='6b Quantized')
eff_ax.plot([key for key in quant_4b_AiQ],[z['net_efficiency'] for z in quant_4b_AiQ.values()], label='4b Quantized')
#eff_ax.plot([key for key in quant_batnorm_AiQ],[z['net_efficiency'] for z in quant_batnorm_AiQ.values()], label='8b Quantized')
eff_ax.legend(loc='best')
eff_plot.savefig(os.path.join(options.outputDir,'t_eff_BOPS.png'))
eff_plot.show()



for label in test_dataset.labels_list:
    aiq_plot = plt.figure()
    aiq_ax = aiq_plot.add_subplot()
    aiq_ax.set_title("{}'s Bkgrd Eff. @ Sig Eff = 50% vs BOPS (HLS4ML Jet Tag)".format(label))
    aiq_ax.grid(True)
    aiq_ax.set_xlabel('Binary Operations (BOPS)')
    aiq_ax.set_ylabel('Background Efficiency')
    aiq_ax.set_yscale('log')
    aiq_ax.set_xscale("log")
    #aiq_ax.plot([key for key in quant_AiQ],[z['sel_bkg_reject'][label] for z in quant_AiQ.values()], label='8b Quant - No BatNorm')
    aiq_ax.plot([key for key in float_AiQ],[z['sel_bkg_reject'][label]  for z in float_AiQ.values()], label='32b Float')
    aiq_ax.plot([key for key in quant_12b_AiQ],[z['sel_bkg_reject'][label]  for z in quant_12b_AiQ.values()], label='12b Quantized')
    aiq_ax.plot([key for key in quant_6b_AiQ],[z['sel_bkg_reject'][label]  for z in quant_6b_AiQ.values()], label='6b Quantized')
    aiq_ax.plot([key for key in quant_4b_AiQ], [z['sel_bkg_reject'][label] for z in quant_4b_AiQ.values()], label='4b Quantized')
    #aiq_ax.plot([key for key in quant_batnorm_AiQ],[z['sel_bkg_reject'][label] for z in quant_batnorm_AiQ.values()], label='8b Quantized')
    aiq_ax.legend(loc='best')
    aiq_plot.savefig(os.path.join(options.outputDir,'bgd_eff_{}_BOPS.png'.format(label)))
    aiq_plot.show()


#Just FC1-3
eff_plot = plt.figure()
eff_ax = eff_plot.add_subplot()
eff_ax.set_title("Efficiency vs BOPS (HLS4ML Jet Tagging Model)")
eff_ax.grid(True)
eff_ax.set_xlabel('Binary Operations (BOPS)')
eff_ax.set_xscale("log")
eff_ax.set_ylabel('Efficiency')
layer_list=['bn1','bn2','bn3', 'fc4']
colors=['red','blue','green', 'orange']
for layer,color in zip(layer_list,colors):
    eff_ax.plot([(model_max_params-key)*32 for key in float_AiQ],
                [z['layer_metrics'][layer]['efficiency'] for z in float_AiQ.values()]
                , label='32b Float - ' + layer + ' Eff.', linestyle='solid', color=color)
    #eff_ax.plot([(model_max_params-key)*8 for key in quant_AiQ],
    #            [z['layer_metrics'][layer]['efficiency']  for z in quant_AiQ.values()]
    #            , label='8b Quant - ' + layer + ' Eff.', linestyle='dashed', color=color)
    eff_ax.plot([key for key in quant_6b_AiQ],
                [z['layer_metrics'][layer]['efficiency'] for z in quant_6b_AiQ.values()]
                , label='6b Quant - ' + layer + ' Eff.', linestyle='dotted', color=color)
    eff_ax.plot([key for key in quant_12b_AiQ],
                [z['layer_metrics'][layer]['efficiency']  for z in quant_12b_AiQ.values()]
                , label='12b Quant - ' + layer + ' Eff.', linestyle='dashdot', color=color)
    eff_ax.plot([key for key in quant_4b_AiQ],
                [z['layer_metrics'][layer]['efficiency']  for z in quant_4b_AiQ.values()]
                , label='4b Quant - ' + layer + ' Eff.', linestyle=(0, (3, 5, 1, 5, 1, 5)), color=color)
  #  eff_ax.plot([key for key in quant_batnorm_AiQ],
  #              [z['layer_metrics'][layer]['efficiency']  for z in quant_batnorm_AiQ.values()]
  #              , label='8b Quant (BN) - ' + layer + ' Eff.', linestyle=(0, (3, 1, 1, 1, 1, 1)), color=color)
eff_ax.legend(loc='best')
eff_plot.savefig(os.path.join(options.outputDir,'layer_eff_BOPS.png'))
eff_plot.show()
