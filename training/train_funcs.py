from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, average_precision_score, auc, accuracy_score
import numpy as np
import torch
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sn
import os.path as path

def l1_regularizer(model, lambda_l1=0.01):
    #  after hours of searching, this man is a god: https://stackoverflow.com/questions/58172188/
    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith('weight'):
            lossl1 += lambda_l1 * model_param_value.abs().sum()
    return lossl1

def train(model, optimizer, loss, train_loader, L1_factor=0.0001, l1reg=True, device='cpu'):
    train_losses = []
    model.to(device)
    model.mask_to_device(device)
    for i, data in enumerate(train_loader, 0):
        local_batch, local_labels = data
        model.train()
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(local_batch.float())
        criterion_loss = loss(outputs, local_labels.float())
        if l1reg:
            reg_loss = l1_regularizer(model, lambda_l1=L1_factor)
        else:
            reg_loss = 0
        total_loss = criterion_loss + reg_loss
        total_loss.backward()
        optimizer.step()
        step_loss = total_loss.item()
        train_losses.append(step_loss)
    return model, train_losses


def val(model, loss, val_loader, L1_factor=0.01, device='cpu'):
    val_roc_auc_scores_list = []
    val_avg_precision_list = []
    val_losses = []
    model.to(device)
    with torch.set_grad_enabled(False):
        model.eval()
        for i, data in enumerate(val_loader, 0):
            local_batch, local_labels = data
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch.float())
            criterion_loss = loss(outputs, local_labels.float())
            reg_loss = l1_regularizer(model, lambda_l1=L1_factor)
            val_loss = criterion_loss + reg_loss
            local_batch, local_labels = local_batch.cpu(), local_labels.cpu()
            outputs = outputs.cpu()
            val_roc_auc_scores_list.append(roc_auc_score(np.nan_to_num(local_labels.numpy()), np.nan_to_num(outputs.numpy())))
            val_avg_precision_list.append(average_precision_score(np.nan_to_num(local_labels.numpy()), np.nan_to_num(outputs.numpy())))
            val_losses.append(val_loss)
    return val_losses, val_avg_precision_list, val_roc_auc_scores_list


def test(model, test_loader, plot=True, nbits=32, outputDir='..', device='cpu', test_dataset_labels=[]):
    #device = torch.device('cpu') #required if doing a untrained init check
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    accuracy_score_value_list = []
    roc_auc_score_list = []
    model.to(device)
    with torch.no_grad():  # Evaulate pruned model performance
        for i, data in enumerate(test_loader):
            model.eval()
            local_batch, local_labels = data
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch.float())
            _, preds = torch.max(outputs, 1)
            predlist = torch.cat([predlist, preds.view(-1).cpu()])
            lbllist = torch.cat([lbllist, torch.max(local_labels, 1)[1].view(-1).cpu()])
        outputs = outputs.cpu()
        local_labels = local_labels.cpu()
        predict_test = outputs.numpy()
        accuracy_score_value_list.append(accuracy_score(np.nan_to_num(lbllist.numpy()), np.nan_to_num(predlist.numpy())))
        roc_auc_score_list.append(roc_auc_score(np.nan_to_num(local_labels.numpy()), np.nan_to_num(outputs.numpy())))

        if plot:
            predict_test = outputs.numpy()
            df = pd.DataFrame()
            fpr = {}
            tpr = {}
            auc1 = {}

            #Time for filenames
            now = datetime.now()
            time = now.strftime("%d-%m-%Y_%H-%M-%S")

            # AUC/Signal Efficiency
            filename = 'ROC_{}b_{}.png'.format(nbits,time)

            sig_eff_plt = plt.figure()
            sig_eff_ax = sig_eff_plt.add_subplot()
            for i, label in enumerate(test_dataset_labels):
                df[label] = local_labels[:, i]
                df[label + '_pred'] = predict_test[:, i]
                fpr[label], tpr[label], threshold = roc_curve(np.nan_to_num(df[label]), np.nan_to_num(df[label + '_pred']))
                auc1[label] = auc(np.nan_to_num(fpr[label]), np.nan_to_num(tpr[label]))
                plt.plot(np.nan_to_num(tpr[label]), np.nan_to_num(fpr[label]),
                         label='%s tagger, AUC = %.1f%%' % (label.replace('j_', ''), np.nan_to_num(auc1[label]) * 100.))
            sig_eff_ax.set_yscale('log')
            sig_eff_ax.set_xlabel("Signal Efficiency")
            sig_eff_ax.set_ylabel("Background Efficiency")
            sig_eff_ax.set_ylim(0.001, 1)
            sig_eff_ax.grid(True)
            sig_eff_ax.legend(loc='upper left')
            sig_eff_plt.savefig(path.join(outputDir, filename))
            sig_eff_plt.show()
            plt.close(sig_eff_plt)

            # Confusion matrix
            filename = 'confMatrix_{}b_{}.png'.format(nbits,time)
            conf_mat = confusion_matrix(np.nan_to_num(lbllist.numpy()), np.nan_to_num(predlist.numpy()))
            df_cm = pd.DataFrame(conf_mat, index=[i for i in test_dataset_labels],
                                 columns=[i for i in test_dataset_labels])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True, fmt='g')
            plt.savefig(path.join(outputDir, filename))
            plt.show()
            plt.close()
    return accuracy_score_value_list, roc_auc_score_list

def test_pruned(model, test_loader, plot=True,
                pruned_params=0, base_params=0, nbits=32,
                outputDir='..', device='cpu', test_dataset_labels=[]):
    #device = torch.device('cpu') #required if doing a untrained init check
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    accuracy_score_value_list = []
    roc_auc_score_list = []
    model.to(device)
    with torch.no_grad():  # Evaulate pruned model performance
        for i, data in enumerate(test_loader):
            model.eval()
            local_batch, local_labels = data
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch.float())
            _, preds = torch.max(outputs, 1)
            predlist = torch.cat([predlist, preds.view(-1).cpu()])
            lbllist = torch.cat([lbllist, torch.max(local_labels, 1)[1].view(-1).cpu()])
        outputs = outputs.cpu()
        local_labels = local_labels.cpu()
        predict_test = outputs.numpy()
        accuracy_score_value_list.append(accuracy_score(np.nan_to_num(lbllist.numpy()), np.nan_to_num(predlist.numpy())))
        roc_auc_score_list.append(roc_auc_score(np.nan_to_num(local_labels.numpy()), np.nan_to_num(outputs.numpy())))

        if plot:
            predict_test = outputs.numpy()
            df = pd.DataFrame()
            fpr = {}
            tpr = {}
            auc1 = {}

            #Time for filenames
            now = datetime.now()
            time = now.strftime("%d-%m-%Y_%H-%M-%S")

            # AUC/Signal Efficiency
            filename = 'ROC_{}b_{}_pruned_{}.png'.format(nbits,pruned_params,time)

            sig_eff_plt = plt.figure()
            sig_eff_ax = sig_eff_plt.add_subplot()
            for i, label in enumerate(test_dataset_labels):
                df[label] = local_labels[:, i]
                df[label + '_pred'] = predict_test[:, i]
                fpr[label], tpr[label], threshold = roc_curve(np.nan_to_num(df[label]), np.nan_to_num(df[label + '_pred']))
                auc1[label] = auc(np.nan_to_num(fpr[label]), np.nan_to_num(tpr[label]))
                plt.plot(np.nan_to_num(tpr[label]), np.nan_to_num(fpr[label]),
                         label='%s tagger, AUC = %.1f%%' % (label.replace('j_', ''), np.nan_to_num(auc1[label]) * 100.))
            sig_eff_ax.set_yscale('log')
            sig_eff_ax.set_xlabel("Signal Efficiency")
            sig_eff_ax.set_ylabel("Background Efficiency")
            sig_eff_ax.set_ylim(0.001, 1)
            sig_eff_ax.grid(True)
            sig_eff_ax.legend(loc='upper left')
            sig_eff_ax.text(0.25, 0.90, '(Pruned {} of {}, {}b)'.format(pruned_params,base_params,nbits),
                        fontweight='bold',
                        wrap=True, horizontalalignment='right', fontsize=12)
            sig_eff_plt.savefig(path.join(outputDir, filename))
            sig_eff_plt.show()
            plt.close(sig_eff_plt)

            # Confusion matrix
            filename = 'confMatrix_{}b_{}_pruned_{}.png'.format(nbits,pruned_params,time)
            conf_mat = confusion_matrix(np.nan_to_num(lbllist.numpy()), np.nan_to_num(predlist.numpy()))
            df_cm = pd.DataFrame(conf_mat, index=[i for i in test_dataset_labels],
                                 columns=[i for i in test_dataset_labels])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True, fmt='g')
            plt.savefig(path.join(outputDir, filename))
            plt.show()
            plt.close()
    return accuracy_score_value_list, roc_auc_score_list