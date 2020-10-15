import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import models
import ae_wav_dataset
import matplotlib.pyplot as plt
from optparse import OptionParser
from sklearn.metrics import mean_squared_error
import torch.optim as optim
import torch.nn.utils.prune as prune
import yaml
import math
import seaborn as sn
import plot_weights
from pytorchtools import EarlyStopping
import copy
from datetime import datetime
import os.path as path
import os
import re
import glob
import itertools

def parse_config(config_file) :
    print("Loading configuration from", config_file)
    config = open(config_file, 'r')
    return yaml.load(config, Loader=yaml.FullLoader)


def countNonZeroWeights(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    return nonzero


def l1_regularizer(model, lambda_l1=0.01):
    #  after hours of searching, this man is a god: https://stackoverflow.com/questions/58172188/
    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith('weight'):
            lossl1 += lambda_l1 * model_param_value.abs().sum()
    return lossl1

def train(model, optimizer, loss, train_loader, L1_factor=0.0001):
    model.to(device)
    model.mask_to_device(device)
    train_losses = []
    for i, data in enumerate(train_loader, 0):
        local_batch, local_labels = data #in AE Wav datset, labels and data are the same thing for convience
        model.train()
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(local_batch.float())
        outputs.to(device)
        criterion_loss = loss(outputs, local_labels.float())
        reg_loss = 0 #Autoencoder doesn't need regularization
        total_loss = criterion_loss + reg_loss
        total_loss.backward()
        optimizer.step()
        step_loss = total_loss.item()
        train_losses.append(step_loss)
    return model, train_losses


def val(model, loss, val_loader, L1_factor=0.01):
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
            val_losses.append(val_loss)
    return val_losses


def test(model, test_loaders, plot=True, pruned_params=0, base_params=0):
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    errors_list = []
    roc_auc_score_list = []
    model.to(device)
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

    return errors_list


def prune_model(model, amount, prune_mask, method=prune.L1Unstructured):
    model.to('cpu')
    model.mask_to_device('cpu')
    for name, module in model.named_modules():  # re-apply current mask to the model
        if isinstance(module, torch.nn.Linear):
            if name is not "dout": #don't prune our final, output layer
                prune.custom_from_mask(module, "weight", prune_mask[name])

    parameters_to_prune = (
        (model.enc1, 'weight'),
        (model.enc2, 'weight'),
        (model.enc3, 'weight'),
        (model.enc4, 'weight'),
        (model.dec1, 'weight'),
        (model.dec2, 'weight'),
        (model.dec3, 'weight'),
        (model.dec4, 'weight'),
        (model.dout, 'weight'),
    )
    prune.global_unstructured(  # global prune the model
        parameters_to_prune,
        pruning_method=method,
        amount=amount,
    )

    for name, module in model.named_modules():  # make pruning "permanant" by removing the orig/mask values from the state dict
        if isinstance(module, torch.nn.Linear):
            if name is not "dout":
                torch.logical_and(module.weight_mask, prune_mask[name],
                                  out=prune_mask[name])  # Update progress mask
                prune.remove(module, 'weight')  # remove all those values in the global pruned model

    return model


def plot_metric_vs_bitparam(model_set,metric_results_set,bit_params_set,base_metrics_set,metric_text):
    # NOTE: Assumes that the first object in the base metrics set is the true base of comparison
    now = datetime.now()
    time = now.strftime("%d-%m-%Y_%H-%M-%S")

    filename = '{}_vs_bitparams'.format(metric_text) + str(time) + '.png'

    for model, metric_results, bit_params in zip(model_set, metric_results_set, bit_params_set):
        nbits = model.weight_precision if hasattr(model, 'weight_precision') else 32
        plt.plot(bit_params, prune_roc_results, linestyle='solid', marker='.', alpha=1, label='Pruned {}b'.format(nbits))

    #Plot "base"/unpruned model points
    for model, base_metric in zip(model_set,base_metrics_set):
        # base_metric = [[num_params],[base_metric]]
        nbits = model.weight_precision if hasattr(model, 'weight_precision') else 32
        plt.plot((base_metric[0] * nbits), (base_metric[1]/base_metrics_set[0][1]), linestyle='solid', marker="X", alpha=1, label='Unpruned {}b'.format(nbits))

    plt.ylabel("{}/FP{}".format(metric_text,metric_text))
    plt.xlabel("Bit Params (Params * bits)")
    plt.grid(color='lightgray', linestyle='-', linewidth=1, alpha=0.3)
    plt.legend(loc='best')
    plt.savefig(path.join(options.outputDir, filename))
    plt.show()


def plot_total_loss(model_set, model_totalloss_set, model_estop_set):
    # Total loss over fine tuning
    now = datetime.now()
    time = now.strftime("%d-%m-%Y_%H-%M-%S")
    for model, model_loss, model_estop in zip(model_set, model_totalloss_set, model_estop_set):
        nbits = model.weight_precision if hasattr(model, 'weight_precision') else 32
        filename = 'total_loss_{}b_{}.png'.format(nbits,time)
        plt.plot(range(1, len(model_loss[0]) + 1), model_loss[0], label='Training Loss')
        plt.plot(range(1, len(model_loss[1]) + 1), model_loss[1], label='Validation Loss')
        # plot each stopping point
        for stop in model_estop:
            plt.axvline(stop, linestyle='--', color='r', alpha=0.3)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.grid(True)
        plt.legend(loc='best')
        plt.title('Total Loss Across pruning & fine tuning {}b model'.format(nbits))
        plt.tight_layout()
        plt.savefig(path.join(options.outputDir,filename))
        plt.show()


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-i','--input'   ,action='store',type='string',dest='inputFile'   ,default='', help='location of data to train off of')
    parser.add_option('-o','--output'   ,action='store',type='string',dest='outputDir'   ,default='train_simple/', help='output directory')
    parser.add_option('-t','--test'   ,action='store',type='string',dest='test'   ,default='', help='Location of test data set')
    parser.add_option('-l','--load', action='store', type='string', dest='modelLoad', default=None, help='Model to load instead of training new')
    parser.add_option('-c','--config'   ,action='store',type='string',dest='config'   ,default='configs/train_config_threelayer.yml', help='tree name')
    parser.add_option('-e','--epochs'   ,action='store',type='int', dest='epochs', default=100, help='number of epochs to train for')
    parser.add_option('-p', '--patience', action='store', type='int', dest='patience', default=10,help='Early Stopping patience in epochs')
    parser.add_option('-L', '--lottery', action='store_true', dest='lottery', default=False, help='Prune and Train using the Lottery Ticket Hypothesis')

    (options,args) = parser.parse_args()
    yamlConfig = parse_config(options.config)
   # prune_value_set = [0.10, 0.111, .125, .143, .166, .20, .25, .333, .50, .666, #take ~10% of the "original" value each time, reducing to ~15% original network size
   #                    0]
    prune_value_set = [0.10, 0.111, .125, .143, .166, .20, .25,  #take ~10% of the "original" value each time to ~70%
                       0] # Last 0 is so the final iteration can fine tune before testing

    prune_mask_set = [
        { #Float 32b
            "enc1":torch.ones(128, 640),
            "enc2":torch.ones(128, 128),
            "enc3":torch.ones(128, 128),
            "enc4":torch.ones(8, 128),
            "dec1":torch.ones(128, 8),
            "dec2":torch.ones(128, 128),
            "dec3":torch.ones(128, 128),
            "dec4":torch.ones(128, 128),
            "dout":torch.ones(640, 128),
        },
        { #Quant 8b
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
    ]
    # First model should be the "Base" model that all other accuracies are compared to!
    model_set = [models.t2_autoencoder_masked(prune_mask_set[0]),
                 models.t2_autoencoder_masked_bv(prune_mask_set[0], 8)]

    # Sets for per-model Results/Data to plot
    prune_result_set = []
    prune_roc_set = []
    bit_params_set = []
    model_totalloss_set = []
    model_estop_set = []

    base_quant_accuracy_score, base_accuracy_score = None, None

    first_run = True
    first_quant = False

    # Setup cuda, TODO CUDA currently not working, investigate why
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Set Batch size and split value
    batch_size = 512 #Keras Model is 512
    train_split = 0.9

    # Setup and split dataset
    full_dataset = ae_wav_dataset.aeWavDataset(options.inputFile)
    #automate creation of each dataset in the future, just pass in the list of dataloaders to the test func, create seperate dataset for that with anomoly tags?
    test_dataset_01 = ae_wav_dataset.aeWavDataset(os.path.join(options.test + "/id_01"))
    test_dataset_03 = ae_wav_dataset.aeWavDataset(os.path.join(options.test + "/id_03"))
    test_dataset_05 = ae_wav_dataset.aeWavDataset(os.path.join(options.test + "/id_05"))
    train_size = int(train_split * len(full_dataset))  # Mimic 90/10 train val split of Keras model
    val_size = len(full_dataset) - train_size
    test_size1 = len(test_dataset_01)
    test_size3 = len(test_dataset_03)
    test_size5 = len(test_dataset_05)

    num_val_batches = math.ceil(val_size/batch_size)
    num_train_batches = math.ceil(train_size/batch_size)
    print("train_batches " + str(num_train_batches))
    print("val_batches " + str(num_val_batches))

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset,[train_size,val_size])

    print("train dataset size: " + str(train_size))
    print("validation dataset size: " + str(val_size))
    print("test id_01 dataset size: " + str(test_size1))
    print("test id_03 dataset size: " + str(test_size3))
    print("test id_05 dataset size: " + str(test_size5))
    print("Result Output Directory set to: {}".format(options.outputDir))

    # Setup dataloaders with our dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=10, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=10, pin_memory=True)
    test_loader_01 = torch.utils.data.DataLoader(test_dataset_01, batch_size=test_size1,
                                              shuffle=True, num_workers=10, pin_memory=True)
    test_loader_03 = torch.utils.data.DataLoader(test_dataset_03, batch_size=test_size3,
                                              shuffle=True, num_workers=10, pin_memory=True)
    test_loader_05 = torch.utils.data.DataLoader(test_dataset_05, batch_size=test_size5,
                                              shuffle=True, num_workers=10, pin_memory=True)

    for model, prune_mask in zip(model_set, prune_mask_set):
        # Model specific results/data to plot
        prune_results = []
        prune_roc_results = []
        bit_params = []
        model_loss = [[], []]  # Train, Val
        model_estop = []
        epoch_counter = 0
        pruned_params = 0
        for prune_value in prune_value_set:
            # Epoch specific plot values
            avg_train_losses = []
            avg_valid_losses = []
            val_roc_auc_scores_list = []
            avg_precision_scores = []
            accuracy_scores = []

            early_stopping = EarlyStopping(patience=options.patience, verbose=True)

            model.update_masks(prune_mask)  # Make sure to update the masks within the model

            optimizer = optim.Adam(model.parameters(), lr=0.001) #Stock DCASE2020 T2 LR is 0.001
            criterion = nn.MSELoss() #Keras model uses MSE

            L1_factor = 0.0001  # Default Keras L1 Loss
            estop = False

            if options.lottery:  # If using lottery ticket method, reset all weights to first initalized vals
                for name, module in model.named_children():  # TODO Not working right now? Need to investigate implementation, dont use
                    if isinstance(module, nn.Linear):
                        print('resetting ', name)
                        # torch.manual_seed(42) #Placeholder for now, dont actually rely on until verified
                        # module.reset_parameters()

            for epoch in range(options.epochs):  # loop over the dataset multiple times
                epoch_counter += 1
                # Train
                model, train_losses = train(model, optimizer, criterion, train_loader, L1_factor=L1_factor)

                # Validate
                val_losses = val(model, criterion, val_loader, L1_factor=L1_factor)

                # Calculate average epoch statistics
                try:
                    train_loss = np.average(train_losses)
                except:
                    train_loss = torch.mean(torch.stack(train_losses)).cpu().numpy()

                try:
                    valid_loss = np.average(val_losses)
                except:
                    valid_loss = torch.mean(torch.stack(val_losses)).cpu().numpy()


                # Print epoch statistics
                print('[epoch %d] train batch loss: %.7f' % (epoch + 1, train_loss))
                print('[epoch %d] val batch loss: %.7f' % (epoch + 1, valid_loss))

                # Check if we need to early stop
                early_stopping(valid_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    estop = True
                    break

            # Load last/best checkpoint model saved via earlystopping
            model.load_state_dict(torch.load('checkpoint.pt'))

            # Time for plots
            now = datetime.now()
            time = now.strftime("%d-%m-%Y_%H-%M-%S")

            # Plot & save losses for this iteration
            plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label='Training Loss')
            plt.plot(range(1, len(avg_valid_losses) + 1), avg_valid_losses, label='Validation Loss')

            # find position of lowest validation loss
            if estop:
                minposs = avg_valid_losses.index(min(avg_valid_losses))
            else:
                minposs = options.epochs-1
            model_loss[0].extend(avg_train_losses[:minposs])
            model_loss[1].extend(avg_valid_losses[:minposs])

            # save position of estop overall app epochs
            model_estop.append(epoch_counter - ((len(avg_valid_losses)) - minposs))

            # update our epoch counter to represent where the model actually stopped training
            epoch_counter -= ((len(avg_valid_losses)) - minposs)

            # Plot losses for this iter
            plt.axvline(minposs+1, linestyle='--', color='r', label='Early Stopping Checkpoint')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.grid(True)
            plt.legend()
            filename = 'loss_plot_e{}_{}_.png'.format(epoch_counter,time)
            plt.savefig(path.join(options.outputDir, filename), bbox_inches='tight')
            plt.show()


            # Prune & Test model
            nbits = model.weight_precision if hasattr(model, 'weight_precision') else 32
            # Time for filenames
            now = datetime.now()
            time = now.strftime("%d-%m-%Y_%H-%M-%S")

            if first_run:
                # Test base model, first iteration of the float model
                print("Base Float Model:")
                base_params = countNonZeroWeights(model)
                MSE_values = test(model, [test_loader_01, test_loader_03, test_loader_05],
                                  pruned_params=0, base_params=base_params)
                base_MSE_average = np.average(MSE_values)
                print('[Base Model] Base MSE (avg): %.7f' % (base_MSE_average))
                filename = path.join(options.outputDir, 'AE_weight_dist_{}b_Base.png'.format(nbits, epoch_counter))
                plot_weights.plot_kernels(model,
                                          text=' (Unpruned FP Model)',
                                          output=filename)
                model_filename = path.join(options.outputDir, "AE_{}b_unpruned_{}.pth".format(nbits, time))
                torch.save(model.state_dict(),model_filename)
                first_run = False
            elif first_quant:
                # Test Unpruned, Base Quant model
                print("Base Quant Model: ")
                base_quant_params = countNonZeroWeights(model)
                MSE_values = test(model, [test_loader_01, test_loader_03, test_loader_05],
                                  pruned_params=0, base_params=base_params)
                base_quant_MSE_average = np.average(MSE_values)
                print('[Base Quant Model] Base MSE (avg): %.7f' % (base_quant_MSE_average))
                filename = path.join(options.outputDir, 'AE_weight_dist_{}b_qBase.png'.format(nbits, epoch_counter))
                plot_weights.plot_kernels(model,
                                          text=' (Unpruned Quant Model)',
                                          output=filename)
                model_filename = path.join(options.outputDir, "AE_{}b_unpruned_{}.pth".format(nbits, time))
                torch.save(model.state_dict(),model_filename)
                first_quant = False
            else:
                print("Pre Pruning:")
                current_params = countNonZeroWeights(model)
                MSE_values = test(model, [test_loader_01,test_loader_03,test_loader_05] , pruned_params=(base_params-current_params), base_params=base_params)
                MSE_average = np.average(MSE_values)
                print('[{}% Pruned Model] MSE (avg): {}'.format( ((current_params/base_params)*100), MSE_average))
                prune_results.append((MSE_average / base_MSE_average))
                bit_params.append(current_params * nbits)
                model_filename = path.join(options.outputDir, "AE_{}b_{}pruned_{}.pth".format(nbits, (base_params-current_params), time))
                torch.save(model.state_dict(),model_filename)

            # Prune for next iter
            if prune_value > 0:
                model = prune_model(model, prune_value, prune_mask)
                # Plot weight dist
                filename = path.join(options.outputDir, 'AE_weight_dist_{}b_e{}_{}.png'.format(nbits, epoch_counter, time))
                print("Post Pruning: ")
                pruned_params = countNonZeroWeights(model)
                plot_weights.plot_kernels(model,
                                          text=' (Pruned ' + str(base_params - pruned_params) + ' out of ' + str(
                                              base_params) + ' params)',
                                          output=filename)

        if not first_quant and base_quant_accuracy_score is None:
            first_quant = True

        bit_params_set.append(bit_params)
        prune_result_set.append(prune_results)
        model_totalloss_set.append(model_loss)
        model_estop_set.append(model_estop)

    # Plot metrics
    base_MSE_set = [[base_params, base_MSE_average],
                    [base_quant_params, base_quant_MSE_average]]


    plot_total_loss(model_set, model_totalloss_set, model_estop_set)
    #plot_metric_vs_bitparam(model_set,prune_result_set,bit_params_set,base_acc_set,metric_text='ACC')
    plot_metric_vs_bitparam(model_set, prune_result_set, bit_params_set, base_MSE_set, metric_text='MSE')