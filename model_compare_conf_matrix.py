from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import sys
import os
import os.path as path
from optparse import OptionParser
import models
import tools.parse_yaml_config
import jet_dataset
import matplotlib.lines as mlines
from tools.param_count import calc_BOPS,countNonZeroWeights

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

def compare(model, model2, test_loader, outputDir='..', device='cpu', test_dataset_labels=[],filetext="", m1txt="", m2txt=""):
    #device = torch.device('cpu') #required if doing a untrained init check
    outlst1 = torch.zeros(0, dtype=torch.long, device='cpu')
    outlst2 = torch.zeros(0, dtype=torch.long, device='cpu')
    accuracy_score_value_list = []
    model.to(device)
    model2.to(device)
    model.mask_to_device(device)
    model2.mask_to_device(device)
    with torch.no_grad():  # Evaulate pruned model performance
        for i, data in enumerate(test_loader):
            model.eval()
            local_batch, local_labels = data
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch.float())
            _, preds = torch.max(outputs, 1)
            outlst1 = torch.cat([outlst1, preds.view(-1).cpu()])
        for i, data in enumerate(test_loader):
            model2.eval()
            local_batch2, local_labels2 = data
            local_batch2, local_labels2 = local_batch2.to(device), local_labels2.to(device)
            outputs2 = model2(local_batch2.float())
            _, preds2 = torch.max(outputs2, 1)
            outlst2 = torch.cat([outlst2, preds2.view(-1).cpu()])

        accuracy_score_value_list.append(accuracy_score(np.nan_to_num(outlst1.numpy()), np.nan_to_num(outlst2.numpy())))

        # Confusion matrix
        filename = 'comp_confMatrix_{}.png'.format(filetext)
        conf_mat = confusion_matrix(np.nan_to_num(outlst1.numpy()), np.nan_to_num(outlst2.numpy()))
        df_cm = pd.DataFrame(conf_mat, index=[i for i in test_dataset_labels],
                             columns=[i for i in test_dataset_labels])
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, fmt='g')
        plt.title("Comparative Model Outputs")
        plt.xlabel(m2txt)
        plt.ylabel(m1txt)
        plt.savefig(path.join(outputDir, filename))
        plt.show()
        plt.close()
    return np.average(accuracy_score_value_list)#, conf_mat

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-i', '--input', action='store', type='string', dest='inputFile', default='../train_data/test/',
                      help='location of data to test off of')
    parser.add_option('-m', '--model_files', action='store', type='string', dest='model_files', default='../model_files/L1BN/',
                      help='Model Files')
    parser.add_option('-o', '--output', action='store', type='string', dest='outputDir', default='results/',
                      help='output directory')
    parser.add_option('-c', '--config', action='store', type='string', dest='config',
                      default=path.join('..','configs','train_config_threelayer.yml'), help='tree name')
    parser.add_option('-a', '--no_bn_affine', action='store_false', dest='bn_affine', default=True,
                      help='disable BN Affine Parameters')
    parser.add_option('-s', '--no_bn_stats', action='store_false', dest='bn_stats', default=True,
                      help='disable BN running statistics')
    parser.add_option('-b', '--no_batnorm', action='store_false', dest='batnorm', default=True,
                      help='disable BatchNormalization (BN) Layers ')

    (options, args) = parser.parse_args()
    yamlConfig = tools.parse_yaml_config.parse_config(options.config)
    prunevals = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '96.6', '98.8']

    markers = ['D', 'x', 'o', 'v', '^', '<', '>', 's', 'p', 'P', '*', 'H', 'h', '+', ',', 'X', 'D', 'd', '|', '_', '1',
               '2', '3', '4', '8', ]
    prunevals2 = ['0.0%', '10.0%', '20.0%', '30.0%', '40.0%', '50.0%', '60.0%', '70.0%', '80.0%', '90.0%', '96.6%',
                 '98.8%']
    marker_lines = []
    rand_vals = [0, 50, 75, 90]
    for mark, pval in zip(markers, reversed(prunevals2)):
        m = mlines.Line2D([], [], color='black', marker=mark, linestyle='None',
                          markersize=10, label=pval)
        marker_lines.append(m)
    if not os.path.exists(options.outputDir):  # create given output directory if it doesnt exist
        os.makedirs(options.outputDir)

    use_cuda = torch.cuda.is_available()
    device = "cpu" # torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Using Device: {}".format(device))
    if use_cuda:
        print("cuda:0 device type: {}".format(torch.cuda.get_device_name(0)))
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(yamlConfig["Seed"])
    torch.cuda.manual_seed_all(yamlConfig["Seed"]) #seeds all GPUs, just in case there's more than one
    np.random.seed(yamlConfig["Seed"])

    test_dataset = jet_dataset.ParticleJetDataset("../train_data/test/", yamlConfig)
    test_size = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size,
                                              shuffle=False, num_workers=10, pin_memory=True)
    test_labels = test_dataset.labels_list
    # For the time being, load our two specific models that we're testing this plot out with. Will flesh out if
    #warrented later on
    model_loc = '../model_files/conf_mat_compare'
    model_loc2 = '../model_files/L1BN/32b'
    loadfile1 = path.join(model_loc,'6b_90rand_80pruned_L1.pth')
    loadfile2 = path.join(model_loc,'6b_90rand_80pruned_noL1.pth')
    baseline_file = path.join(model_loc,'32b_unpruned_25-11-2020_02-37-19.pth')
    standard_mask = {
        "fc1": torch.ones(64, 16),
        "fc2": torch.ones(32, 64),
        "fc3": torch.ones(32, 32),
        "fc4": torch.ones(5, 32)}

    m1 = models.three_layer_model_bv_batnorm_masked(masks=standard_mask, precision=6)
    m1.load_state_dict(torch.load(os.path.join(loadfile1), map_location=device))

    m2 = models.three_layer_model_bv_batnorm_masked(masks=standard_mask, precision=6)
    m2.load_state_dict(torch.load(os.path.join(loadfile2), map_location=device))

    baseline = models.three_layer_model_batnorm_masked(masks=standard_mask, bn_affine=options.bn_affine,
                                                            bn_stats=options.bn_stats)
    baseline.load_state_dict(torch.load(os.path.join(baseline_file), map_location=device))

    #c_acc = compare(m1,m2,test_loader,options.outputDir,device=device,test_dataset_labels=test_labels,
    #                      filetext="NoL1_vs_L1_90Rand80Prune",
    #                      m1txt="6b, 90% Rand, 80% Pruned w/L1", m2txt="6b, 90% Rand, 80% Pruned no L1")
    #print("Relative Perf - Accuracy: {}".format(c_acc))
    #print("Relative Perf - Conf Matrix:")
    #print(c_cm)


    #  dir = "model_files/"
    dir = options.model_files

    use_cuda = torch.cuda.is_available()
    device = "cpu" # torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Using Device: {}".format(device))
    if use_cuda:
        print("cuda:0 device type: {}".format(torch.cuda.get_device_name(0)))

    try:
        if options.batnorm:
            loadmodel = models.three_layer_model_batnorm_masked(standard_mask, bn_affine=options.bn_affine,
                                                            bn_stats=options.bn_stats)
        else:
            loadmodel = models.three_layer_model_masked(standard_mask)  # 32b

        float_model_set, model_max_params = gen_model_dict(loadmodel, os.path.join(dir, '32b'))
    except Exception as e:
        print(e)

    try:
        if options.batnorm:
            loadmodel = models.three_layer_model_bv_batnorm_masked(standard_mask,6, bn_affine=options.bn_affine, bn_stats=options.bn_stats)
        else:
            loadmodel = models.three_layer_model_bv_masked(standard_mask, 6)

        quant_model_set_6b, quant_6b_max_params = gen_model_dict(loadmodel, os.path.join(dir, '6b'))
    except Exception as e:
        print(e)

    float_AiQ = {}
    for (model_bops, model_file), pruneval in zip(sorted(float_model_set.items()),reversed(prunevals)):
        if options.batnorm:
            loadmodel = models.three_layer_model_batnorm_masked(standard_mask, bn_affine=options.bn_affine,
                                                                bn_stats=options.bn_stats)
        else:
            loadmodel = models.three_layer_model_masked(standard_mask)  # 32b
        loadmodel.load_state_dict(torch.load(os.path.join(dir, '32b', model_file), map_location=device))
        print('Calculating AiQ for 32b, ' + str(model_bops) + ' BOPS')
        float_AiQ.update({model_bops: compare(baseline,loadmodel,test_loader,options.outputDir,device='cpu',test_dataset_labels=test_labels,
                          filetext="32b0Prune_vs_32b{}prune_L1BN".format(pruneval),
                          m1txt="32b,0% Pruned w/L1+BN", m2txt="32b, {}% Pruned, L1+BN".format(pruneval))})

    quant_6b_AiQ = {}
    for (model_bops, model_file), pruneval in zip(sorted(quant_model_set_6b.items()),reversed(prunevals)):
        if options.batnorm:
            loadmodel = models.three_layer_model_bv_batnorm_masked(standard_mask, 6, bn_affine=options.bn_affine,
                                                                   bn_stats=options.bn_stats)
        else:
            loadmodel = models.three_layer_model_bv_masked(standard_mask, 6)
        loadmodel.load_state_dict(torch.load(os.path.join(dir, '6b', model_file), map_location=device))
        print('Calculating AiQ for 6b, ' + str(model_bops) + ' BOPS')
        quant_6b_AiQ.update({model_bops: compare(baseline,loadmodel,test_loader,options.outputDir,device='cpu',test_dataset_labels=test_labels,
                          filetext="32b0Prune_vs_6b{}prune_L1BN".format(pruneval),
                          m1txt="32b,0% Pruned w/L1+BN", m2txt="6b, {}% Pruned, L1+BN".format(pruneval))})


    # Accuracy plot
    eff_plot = plt.figure()
    eff_ax = eff_plot.add_subplot()
    eff_ax.grid(True)
    eff_ax.set_xlabel('BOPs')
    eff_ax.set_ylabel('Relative Accuracy')
    eff_ax.set_xscale("symlog", linthresh=1e6)
    eff_ax.set_xlim(1e4, 1e7)
    eff_ax.set_ylim(0.8, 1)
    eff_ax.set_title("Relative Accuracy vs BOPS (Relative to 32b 0% Prune, L1+BN)")
    eff_ax.plot([int(key) for key in float_AiQ], [z for z in float_AiQ.values()],
                label='32b FT w/ L1 Reg & BN',
                linestyle='solid', color='red')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in float_AiQ],
                          [z for z in float_AiQ.values()], markers):
        eff_ax.plot(x, y, linestyle='solid', marker=mark, markersize=6, color='red', label='_nolegend_')

    eff_ax.plot([int(key) for key in quant_6b_AiQ], [z for z in quant_6b_AiQ.values()],
                label='6b FT w/ L1 Reg & BN',
                linestyle='dotted', color='red')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in quant_6b_AiQ],
                          [z for z in quant_6b_AiQ.values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=6, color='red', label='_nolegend_')

    eff_ax.add_artist(
        plt.legend(handles=eff_ax.get_lines(), title='Fine tuning pruning', loc='lower left', framealpha=0.5))
    #eff_ax.add_artist(
    #    plt.legend(handles=marker_lines, title='Percent pruned (approx.)', loc='upper right', framealpha=0.5))
    eff_plot.savefig(path.join(options.outputDir, 'Rel_ACC_FT_32_6_L1_zoom.png'))
    # eff_plot.savefig('ACC_FT_BNcomp.pdf')
    eff_plot.show()