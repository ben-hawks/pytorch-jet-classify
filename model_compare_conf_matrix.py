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


def compare(model, model2, test_loader, outputDir='..', device='cpu', test_dataset_labels=[],filetext="", m1txt="", m2txt=""):
    #device = torch.device('cpu') #required if doing a untrained init check
    outlst1 = torch.zeros(0, dtype=torch.long, device='cpu')
    outlst2 = torch.zeros(0, dtype=torch.long, device='cpu')
    accuracy_score_value_list = []
    model.to(device)
    model2.to(device)
    with torch.no_grad():  # Evaulate pruned model performance
        for i, data in enumerate(test_loader):
            model.eval()
            local_batch, local_labels = data
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch.float())
            outputs2 = model2(local_batch.float())
            _, preds = torch.max(outputs, 1)
            _, preds2 = torch.max(outputs2, 1)
            outlst1 = torch.cat([outlst1, preds.view(-1).cpu()])
            outlst2 = torch.cat([outlst2, preds2.view(-1).cpu()])
        accuracy_score_value_list.append(accuracy_score(np.nan_to_num(outlst1.numpy()), np.nan_to_num(outlst2.numpy())))

        # Confusion matrix
        filename = 'comp_confMatrix_{}.png'.format(filetext)
        conf_mat = confusion_matrix(np.nan_to_num(outlst1.numpy()), np.nan_to_num(outlst2.numpy()), normalize=True)
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
    return np.average(accuracy_score_value_list), conf_mat

if __name__ == "__main__":
        parser = OptionParser()
        parser.add_option('-i', '--input', action='store', type='string', dest='inputFile', default='train_data/test',
                          help='location of data to test off of')
        parser.add_option('-o', '--output', action='store', type='string', dest='outputDir', default='results/',
                          help='output directory')
        parser.add_option('-c', '--config', action='store', type='string', dest='config',
                          default='configs/train_config_threelayer.yml', help='tree name')
        (options, args) = parser.parse_args()
        yamlConfig = tools.parse_yaml_config.parse_config(options.config)

        if not os.path.exists(options.outputDir):  # create given output directory if it doesnt exist
            os.makedirs(options.outputDir)

        use_cuda = torch.cuda.is_available()
        device = "cpu" # torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        print("Using Device: {}".format(device))
        if use_cuda:
            print("cuda:0 device type: {}".format(torch.cuda.get_device_name(0)))

        test_dataset = jet_dataset.ParticleJetDataset(options.inputFile, yamlConfig)
        test_size = len(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size,
                                                  shuffle=False, num_workers=10, pin_memory=True)
        test_labels = test_dataset.labels_list
        # For the time being, load our two specific models that we're testing this plot out with. Will flesh out if
        #warrented later on
        model_loc = 'model_files/conf_mat_compare'
        loadfile1 = path.join(model_loc,'6b_90rand_80pruned_L1.pth')
        loadfile2 = path.join(model_loc,'6b_90rand_80pruned_noL1.pth')

        standard_mask = {
            "fc1": torch.ones(64, 16),
            "fc2": torch.ones(32, 64),
            "fc3": torch.ones(32, 32),
            "fc4": torch.ones(5, 32)}

        m1 = models.three_layer_model_bv_batnorm_masked(masks=standard_mask, precision=6)
        m1.load_state_dict(torch.load(os.path.join(loadfile1), map_location=device))

        m2 = models.three_layer_model_bv_batnorm_masked(masks=standard_mask, precision=6)
        m2.load_state_dict(torch.load(os.path.join(loadfile2), map_location=device))

        c_acc, c_cm = compare(m1,m2,test_loader,options.outputDir,device=device,test_dataset_labels=test_labels,
                              filetext="NoL1_vs_L1_90Rand80Prune",
                              m1txt="6b, 90% Rand, 80% Pruned w/L1", m2txt="6b, 90% Rand, 80% Pruned no L1")
        print("Relative Perf - Accuracy: {}".format(c_acc))
        print("Relative Perf - Conf Matrix:")
        print(c_cm)
