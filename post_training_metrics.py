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
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

from optparse import OptionParser
from tools.parse_yaml_config import parse_config
from tools.aiq import calc_AiQ
from tools.param_count import countNonZeroWeights, calc_BOPS

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
    # generates dictionary of models and according to their BOPS values, given a folder containing the model files
    # Provide an instance of the model you're loading (to calculate # params)
    # and the path to folder containing model files, returns a dict with the format {BOPS:path to model}
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
    else:
        loadmodel = models.three_layer_model_masked(prune_mask_set)  # 32b

    float_model_set, model_max_params = gen_model_dict(loadmodel, os.path.join(dir, '32b'))
except Exception as e:
    print(e)
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
        loadmodel = models.three_layer_model_masked(prune_mask_set)  # 32b

    print('Calculating AiQ for 32b, ' + str(model_bops) + ' BOPS')
    float_AiQ.update({model_bops: calc_AiQ(loadmodel, test_loader, loadfile=os.path.join(dir, '32b', model_file), batnorm = options.batnorm, device='cpu')})

quant_4b_AiQ = {}
for model_bops, model_file in sorted(quant_model_set_4b.items()):
    if options.batnorm:
        loadmodel = models.three_layer_model_bv_batnorm_masked(prune_mask_set,4, bn_affine=options.bn_affine, bn_stats=options.bn_stats)
    else:
        loadmodel = models.three_layer_model_bv_masked(prune_mask_set, 4)
    print('Calculating AiQ for 4b, ' + str(model_bops) + ' BOPS')
    quant_4b_AiQ.update({model_bops: calc_AiQ(loadmodel, test_loader, loadfile=os.path.join(dir, '4b', model_file), batnorm = options.batnorm, device='cpu')})

quant_6b_AiQ = {}
for model_bops, model_file in sorted(quant_model_set_6b.items()):
    if options.batnorm:
        loadmodel = models.three_layer_model_bv_batnorm_masked(prune_mask_set,6, bn_affine=options.bn_affine, bn_stats=options.bn_stats)
    else:
        loadmodel = models.three_layer_model_bv_masked(prune_mask_set, 6)
    print('Calculating AiQ for 6b, ' + str(model_bops) + ' BOPS')
    quant_6b_AiQ.update({model_bops: calc_AiQ(loadmodel, test_loader, loadfile=os.path.join(dir, '6b', model_file), batnorm = options.batnorm, device='cpu')})

quant_12b_AiQ = {}
for model_bops, model_file in sorted(quant_model_set_12b.items()):
    if options.batnorm:
        loadmodel = models.three_layer_model_bv_batnorm_masked(prune_mask_set,12, bn_affine=options.bn_affine, bn_stats=options.bn_stats)
    else:
        loadmodel = models.three_layer_model_bv_masked(prune_mask_set, 12)
    print('Calculating AiQ for 12b, ' + str(model_bops) + ' BOPS')
    quant_12b_AiQ.update({model_bops: calc_AiQ(loadmodel, test_loader, loadfile=os.path.join(dir, '12b', model_file), batnorm = options.batnorm, device='cpu')})

quant_batnorm_AiQ = {}
for model_bops, model_file in sorted(quant_batnorm_model_set.items()):
    if options.batnorm:
        loadmodel = models.three_layer_model_bv_batnorm_masked(prune_mask_set,8, bn_affine=options.bn_affine, bn_stats=options.bn_stats)
    else:
        loadmodel = models.three_layer_model_bv_masked(prune_mask_set, 8)
    print('Calculating AiQ for 8b w/ BatchNorm, ' + str(model_bops) + ' BOPS')
    quant_batnorm_AiQ.update({model_bops: calc_AiQ(loadmodel, test_loader, loadfile=os.path.join(dir, '8b', model_file), batnorm = options.batnorm, device='cpu')})

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
if options.batnorm:
    layer_list=['bn1','bn2','bn3','fc4']
else:
    layer_list=['fc1','fc3','fc3','fc4']
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
