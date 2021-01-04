# Import misc packages
import math
import json
import os
import os.path as path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from optparse import OptionParser

# Import torch stuff
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.prune as prune

# Import our own code
import models
import jet_dataset
from training.early_stopping import EarlyStopping
from training.train_funcs import train, val, test_pruned as test
from tools.aiq import calc_AiQ 
from training.training_plots import plot_total_loss, plot_total_eff, plot_metric_vs_bitparam, plot_kernels
from tools.param_count import countNonZeroWeights
from tools.parse_yaml_config import parse_config

def prune_model(model, amount, prune_mask, method=prune.L1Unstructured):
    model.to('cpu')
    model.mask_to_device('cpu')
    for name, module in model.named_modules():  # re-apply current mask to the model
        if isinstance(module, torch.nn.Linear):
#            if name is not "fc4":
             prune.custom_from_mask(module, "weight", prune_mask[name])

    parameters_to_prune = (
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
        (model.fc3, 'weight'),
        (model.fc4, 'weight'),
    )
    prune.global_unstructured(  # global prune the model
        parameters_to_prune,
        pruning_method=method,
        amount=amount,
    )

    for name, module in model.named_modules():  # make pruning "permanant" by removing the orig/mask values from the state dict
        if isinstance(module, torch.nn.Linear):
#            if name is not "fc4":
            torch.logical_and(module.weight_mask, prune_mask[name],
                              out=prune_mask[name])  # Update progress mask
            prune.remove(module, 'weight')  # remove all those values in the global pruned model

    return model


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
    parser.add_option('-a', '--no_bn_affine', action='store_false', dest='bn_affine', default=True, help='disable BN Affine Parameters')
    parser.add_option('-s', '--no_bn_stats', action='store_false', dest='bn_stats', default=True, help='disable BN running statistics')
    parser.add_option('-b', '--no_batnorm', action='store_false', dest='batnorm', default=True, help='disable BatchNormalization (BN) Layers ')
    parser.add_option('-r', '--no_l1reg', action='store_false', dest='l1reg', default=True, help='disable L1 Regularization totally ')
    parser.add_option('-m', '--model_set', type='str', dest='model_set', default='32,12,8,6,4', help='comma separated list of which bit widths to run')
    parser.add_option('-n', '--net_efficiency', action='store_true', dest='efficiency_calc', default=False, help='Enable Per-Epoch efficiency calculation (adds train time)')
    parser.add_option('-k', '--kfold', action='store', type='int', dest='fold', default=None, help='Which fold to use as a validation fold, other folds being training set')
    parser.add_option('-f', '--folds', action='store', type='int', dest='kfolds', default=4, help='K Folds, number of total folds')
    (options,args) = parser.parse_args()
    yamlConfig = parse_config(options.config)

    # create given output directory if it doesnt exist
    if not path.exists(options.outputDir):
        os.makedirs(options.outputDir, exist_ok=True)

    # take ~10% of the "original" value each time, until last few iterations, reducing to ~1.2% original network size
    prune_value_set = [0.10, 0.111, .125, .143, .166, .20, .25, .333, .50, .666, .666]
    prune_value_set.append(0)  # Last 0 is so the final iteration can fine tune before testing

    standard_mask = {
            "fc1": torch.ones(64, 16),
            "fc2": torch.ones(32, 64),
            "fc3": torch.ones(32, 32),
            "fc4": torch.ones(5, 32)}

    prune_mask_set = [standard_mask for m in range(0, 5)]

    print("Made {} Masks!".format(len(prune_mask_set)))

    #If we're Lottery Ticket Hypothesis Pruning (LT/LTH), fix our seeds
    if options.lottery:
        # fix seed
        torch.manual_seed(yamlConfig["Seed"])
        torch.cuda.manual_seed_all(yamlConfig["Seed"]) #seeds all GPUs, just in case there's more than one
        np.random.seed(yamlConfig["Seed"])

    # First model should be the "Base" model that all other accuracies are compared to!
    if options.batnorm:
        models = {'32': models.three_layer_model_batnorm_masked(prune_mask_set[0], bn_affine=options.bn_affine, bn_stats=options.bn_stats), #32b
                  '12': models.three_layer_model_bv_batnorm_masked(prune_mask_set[1],12, bn_affine=options.bn_affine, bn_stats=options.bn_stats), #12b
                  '8': models.three_layer_model_bv_batnorm_masked(prune_mask_set[2],8, bn_affine=options.bn_affine, bn_stats=options.bn_stats), #8b
                  '6':  models.three_layer_model_bv_batnorm_masked(prune_mask_set[3],6, bn_affine=options.bn_affine, bn_stats=options.bn_stats), #6b
                  '4': models.three_layer_model_bv_batnorm_masked(prune_mask_set[4],4, bn_affine=options.bn_affine, bn_stats=options.bn_stats) #4b
                  }
    else:
        models = {'32': models.three_layer_model_masked(prune_mask_set[0]), #32b
                  '12': models.three_layer_model_bv_masked(prune_mask_set[1],12), #12b
                  '8': models.three_layer_model_bv_masked(prune_mask_set[2],8), #8b
                  '6': models.three_layer_model_bv_masked(prune_mask_set[3],6), #6b
                  '4': models.three_layer_model_bv_masked(prune_mask_set[4],4) #4b
        }

    model_set = [models[m] for m in options.model_set.split(',')]

    #save initalizations in case we're doing Lottery Ticket
    inital_models_sd = []
    for model in model_set:
        inital_models_sd.append(model.state_dict())


    print("# Models to train: {}".format(len(model_set)))
    # Sets for per-model Results/Data to plot
    prune_result_set = []
    prune_roc_set = []
    bit_params_set = []
    model_totalloss_set = []
    model_estop_set = []
    model_eff_set = []
    model_totalloss_json_dict = {}
    model_eff_json_dict = {}
    base_quant_accuracy_score, base_accuracy_score = None, None

    first_run = True
    first_quant = False

    # Setup cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Using Device: {}".format(device))
    if use_cuda:
        print("cuda:0 device type: {}".format(torch.cuda.get_device_name(0)))

    if options.lottery:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.fastest = True

    # Set Batch size and split value
    batch_size = 1024

    if options.fold is None: #No fold passed, just load a whole folder and randomly split train/test
        train_split = 0.75

        # Setup and split dataset
        full_dataset = jet_dataset.ParticleJetDataset(options.inputFile,yamlConfig)

        train_size = int(train_split * len(full_dataset))  # 25% for Validation set, 75% for train set

        val_size = len(full_dataset) - train_size


        num_val_batches = math.ceil(val_size/batch_size)
        num_train_batches = math.ceil(train_size/batch_size)
        print("train_batches " + str(num_train_batches))
        print("val_batches " + str(num_val_batches))

        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset,[train_size,val_size])
    else:
        train_filenames = []
        val_filename = ""
        for i in range(1, options.kfolds+1):
            if i is not options.fold:
                train_filenames.append("jetImage_kfold_{}.h5".format(i))
            else:
                val_filename = "jetImage_kfold_{}.h5".format(i)
        print("K Fold Train Dataset:")
        train_dataset = jet_dataset.ParticleJetDataset(options.inputFile, yamlConfig, filenames=train_filenames)
        print("K Fold Val Dataset:")
        val_dataset = jet_dataset.ParticleJetDataset(options.inputFile, yamlConfig, filenames=[val_filename])

        train_size = int(len(train_dataset))  # 25% for Validation set, 75% for train set
        val_size = int(len(val_dataset))

        num_val_batches = math.ceil(val_size / batch_size)
        num_train_batches = math.ceil(train_size / batch_size)
        print("train_batches " + str(num_train_batches))
        print("val_batches " + str(num_val_batches))

    test_dataset = jet_dataset.ParticleJetDataset(options.test, yamlConfig)
    test_size = len(test_dataset)

    print("train dataset size: " + str(len(train_dataset)))
    print("validation dataset size: " + str(len(val_dataset)))
    print("test dataset size: " + str(len(test_dataset)))


    # Setup dataloaders with our dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=10, pin_memory=True)  # FFS, have to use numworkers = 0 because apparently h5 objects can't be pickled, https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch/issues/69

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=10, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size,
                                              shuffle=False, num_workers=10, pin_memory=True)
    base_quant_params = None

    for model, prune_mask, init_sd in zip(model_set, prune_mask_set, inital_models_sd):
        # Model specific results/data to plot
        prune_results = []
        prune_roc_results = []
        bit_params = []
        model_loss = [[], []]  # Train, Val
        model_estop = []
        model_eff = []
        epoch_counter = 0
        pruned_params = 0
        nbits = model.weight_precision if hasattr(model, 'weight_precision') else 32
        last_stop = 0

        print("~!~!~!~!~!~!~!! Starting Train/Prune Cycle for {}b model! !!~!~!~!~!~!~!~".format(nbits))
        for prune_value in prune_value_set:
            # Epoch specific plot values
            avg_train_losses = []
            avg_valid_losses = []
            val_roc_auc_scores_list = []
            avg_precision_scores = []
            accuracy_scores = []
            iter_eff = []
            early_stopping = EarlyStopping(patience=options.patience, verbose=True)

            model.update_masks(prune_mask)  # Make sure to update the masks within the model

            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            criterion = nn.BCELoss()

            L1_factor = 0.0001  # Default Keras L1 Loss
            estop = False

            if options.efficiency_calc and epoch_counter == 0:  # Get efficiency of un-initalized model
                aiq_dict, aiq_time = calc_AiQ(model,test_loader,batnorm=options.batnorm,device=device)
                epoch_eff = aiq_dict['net_efficiency']
                iter_eff.append(aiq_dict)
                model_estop.append(epoch_counter)
                print('[epoch 0] Model Efficiency: %.7f' % epoch_eff)
                for layer in aiq_dict["layer_metrics"]:
                    print('[epoch 0]\t Layer %s Efficiency: %.7f' % (layer, aiq_dict['layer_metrics'][layer]['efficiency']))

            if options.lottery:  # If using lottery ticket method, reset all weights to first initalized vals
                print("~~~~~!~!~!~!~!~!~Resetting Model!~!~!~!~!~!~~~~~\n\n")
                print("Resetting Model to Inital State dict with masks applied. Verifying via param count.\n\n")
                model.load_state_dict(init_sd)
                model.update_masks(prune_mask)
                model.mask_to_device(device)
                model.force_mask_apply()
                countNonZeroWeights(model)

            for epoch in range(options.epochs):  # loop over the dataset multiple times
                epoch_counter += 1
                # Train
                model, train_losses = train(model, optimizer, criterion, train_loader, L1_factor=L1_factor, l1reg=options.l1reg, device=device)

                # Validate
                val_losses, val_avg_precision_list, val_roc_auc_scores_list = val(model, criterion, val_loader, L1_factor=L1_factor, device=device)

                # Calculate average epoch statistics
                try:
                    train_loss = np.average(train_losses)
                except:
                    train_loss = torch.mean(torch.stack(train_losses)).cpu().numpy()

                try:
                    valid_loss = np.average(val_losses)
                except:
                    valid_loss = torch.mean(torch.stack(val_losses)).cpu().numpy()

                val_roc_auc_score = np.average(val_roc_auc_scores_list)
                val_avg_precision = np.average(val_avg_precision_list)

                if options.efficiency_calc:
                    aiq_dict, aiq_time = calc_AiQ(model,test_loader,batnorm=options.batnorm,device=device)
                    epoch_eff = aiq_dict['net_efficiency']
                    iter_eff.append(aiq_dict)

                avg_train_losses.append(train_loss.tolist())
                avg_valid_losses.append(valid_loss.tolist())
                avg_precision_scores.append(val_avg_precision)

                # Print epoch statistics
                print('[epoch %d] train batch loss: %.7f' % (epoch + 1, train_loss))
                print('[epoch %d] val batch loss: %.7f' % (epoch + 1, valid_loss))
                print('[epoch %d] val ROC AUC Score: %.7f' % (epoch + 1, val_roc_auc_score))
                print('[epoch %d] val Avg Precision Score: %.7f' % (epoch + 1, val_avg_precision))
                if options.efficiency_calc:
                    print('[epoch %d] aIQ Calc Time: %.7f seconds' % (epoch + 1, aiq_time))
                    print('[epoch %d] Model Efficiency: %.7f' % (epoch + 1, epoch_eff))
                    for layer in aiq_dict["layer_metrics"]:
                        print('[epoch %d]\t Layer %s Efficiency: %.7f' % (epoch + 1, layer, aiq_dict['layer_metrics'][layer]['efficiency']))
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
            loss_plt = plt.figure()
            loss_ax = loss_plt.add_subplot()

            loss_ax.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label='Training Loss')
            loss_ax.plot(range(1, len(avg_valid_losses) + 1), avg_valid_losses, label='Validation Loss')

            # find position of lowest validation loss
            if estop:
                minposs = avg_valid_losses.index(min(avg_valid_losses))
            else:
                minposs = options.epochs
            model_loss[0].extend(avg_train_losses[:minposs])
            model_loss[1].extend(avg_valid_losses[:minposs])
            model_eff.extend(iter_eff[:minposs])

            # save position of estop overall app epochs
            model_estop.append(epoch_counter - ((len(avg_valid_losses)) - minposs))


            # update our epoch counter to represent where the model actually stopped training
            epoch_counter -= ((len(avg_valid_losses)) - minposs)

            nbits = model.weight_precision if hasattr(model, 'weight_precision') else 32
            # Plot losses for this iter

            loss_ax.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
            loss_ax.set_xlabel('epochs')
            loss_ax.set_ylabel('loss')
            loss_ax.grid(True)
            loss_ax.legend()
            filename = 'loss_plot_{}b_e{}_{}_.png'.format(nbits,epoch_counter,time)
            loss_ax.set_title('Loss from epoch {} to {}, {}b model'.format(last_stop,epoch_counter,nbits))
            loss_plt.savefig(path.join(options.outputDir, filename), bbox_inches='tight')
            loss_plt.show()
            plt.close(loss_plt)
            if options.efficiency_calc:
                # Plot & save eff for this iteration
                loss_plt = plt.figure()
                loss_ax = loss_plt.add_subplot()
                loss_ax.set_title('Net Eff. from epoch {} to {}, {}b model'.format(last_stop+1, epoch_counter, nbits))
                loss_ax.plot(range(last_stop+1, len(iter_eff) + last_stop+1), [z['net_efficiency'] for z in iter_eff], label='Net Efficiency', color='green')

                #loss_ax.plot(range(1, len(iter_eff) + 1), [z["layer_metrics"][layer]['efficiency'] for z in iter_eff])
                loss_ax.axvline(last_stop+minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
                loss_ax.set_xlabel('epochs')
                loss_ax.set_ylabel('Net Efficiency')
                loss_ax.grid(True)
                loss_ax.legend()
                filename = 'eff_plot_{}b_e{}_{}_.png'.format(nbits,epoch_counter,time)
                loss_plt.savefig(path.join(options.outputDir, filename), bbox_inches='tight')
                loss_plt.show()
                plt.close(loss_plt)

            # Prune & Test model
            last_stop = epoch_counter - ((len(avg_valid_losses)) - minposs)

            # Time for filenames
            now = datetime.now()
            time = now.strftime("%d-%m-%Y_%H-%M-%S")

            if first_run:
                # Test base model, first iteration of the float model
                print("Base Float Model:")
                base_params,_,_,_ = countNonZeroWeights(model)
                accuracy_score_value_list, roc_auc_score_list = test(model, test_loader, pruned_params=0,
                                                                            base_params=base_params, nbits=nbits, 
                                                                            outputDir=options.outputDir, device=device,test_dataset_labels=test_dataset.labels_list)
                base_accuracy_score = np.average(accuracy_score_value_list)
                base_roc_score = np.average(roc_auc_score_list)
                filename = path.join(options.outputDir, 'weight_dist_{}b_Base_{}.png'.format(nbits, time))
                plot_kernels(model, text=' (Unpruned FP Model)', output=filename)
                if not path.exists(path.join(options.outputDir,'models','{}b'.format(nbits))):
                    os.makedirs(path.join(options.outputDir,'models','{}b'.format(nbits)))
                model_filename = path.join(options.outputDir,'models','{}b'.format(nbits), "{}b_unpruned_{}.pth".format(nbits, time))
                torch.save(model.state_dict(),model_filename)
                first_run = False
            elif first_quant:
                # Test Unpruned, Base Quant model
                print("Base Quant Model: ")
                base_quant_params,_,_,_ = countNonZeroWeights(model)
                accuracy_score_value_list, roc_auc_score_list = test(model, test_loader, pruned_params=0,
                                                                            base_params=base_params, nbits=nbits, 
                                                                            outputDir=options.outputDir, device=device,test_dataset_labels=test_dataset.labels_list)
                base_quant_accuracy_score = np.average(accuracy_score_value_list)
                base_quant_roc_score = np.average(roc_auc_score_list)
                filename = path.join(options.outputDir, 'weight_dist_{}b_qBase_{}.png'.format(nbits, time))
                plot_kernels(model, text=' (Unpruned Quant Model)', output=filename)
                if not path.exists(path.join(options.outputDir,'models','{}b'.format(nbits))):
                    os.makedirs(path.join(options.outputDir,'models','{}b'.format(nbits)))
                model_filename = path.join(options.outputDir,'models','{}b'.format(nbits), "{}b_unpruned_{}.pth".format(nbits, time))
                torch.save(model.state_dict(),model_filename)
                first_quant = False
            else:
                print("Pre Pruning:")
                current_params,_,_,_ = countNonZeroWeights(model)
                accuracy_score_value_list, roc_auc_score_list = test(model, test_loader, pruned_params=0,
                                                                            base_params=base_params, nbits=nbits, 
                                                                            outputDir=options.outputDir, device=device,test_dataset_labels=test_dataset.labels_list)
                accuracy_score_value = np.average(accuracy_score_value_list)
                roc_auc_score_value = np.average(roc_auc_score_list)
                prune_results.append(1 / (accuracy_score_value / base_accuracy_score))
                prune_roc_results.append(1/ (roc_auc_score_value/ base_roc_score))
                bit_params.append(current_params * nbits)
                if not path.exists(path.join(options.outputDir,'models','{}b'.format(nbits))):
                    os.makedirs(path.join(options.outputDir,'models','{}b'.format(nbits)))
                model_filename = path.join(options.outputDir,'models','{}b'.format(nbits),"{}b_{}pruned_{}.pth".format(nbits, (base_params-current_params), time))
                torch.save(model.state_dict(),model_filename)

            # Prune for next iter
            if prune_value > 0:
                model = prune_model(model, prune_value, prune_mask)
                # Plot weight dist
                filename = path.join(options.outputDir, 'weight_dist_{}b_e{}_{}.png'.format(nbits, epoch_counter, time))
                print("Post Pruning: ")
                pruned_params,_,_,_ = countNonZeroWeights(model)
                plot_kernels(model,
                                            text=' (Pruned ' + str(base_params - pruned_params) + ' out of ' + str(
                                              base_params) + ' params)',
                                            output=filename)

        if not first_quant and base_quant_accuracy_score is None:
            first_quant = True

        bit_params_set.append(bit_params)
        prune_result_set.append(prune_results)
        prune_roc_set.append(prune_roc_results)
        model_totalloss_set.append(model_loss)
        model_estop_set.append(model_estop)
        model_eff_set.append(model_eff)
        model_totalloss_json_dict.update({nbits:[model_loss,model_eff,model_estop]})

    filename = 'model_losses_{}.json'.format(options.model_set.replace(",","_"))
    with open(os.path.join(options.outputDir, filename), 'w') as fp:
        json.dump(model_totalloss_json_dict, fp)

    if base_quant_params == None:
        base_acc_set = [[base_params, base_accuracy_score]]
        base_roc_set = [[base_params, base_roc_score]]
    else:

        base_acc_set = [[base_params, base_accuracy_score],
                        [base_quant_params, base_quant_accuracy_score]]

        base_roc_set = [[base_params, base_roc_score],
                        [base_quant_params, base_quant_roc_score]]
    # Plot metrics
    plot_total_loss(model_set, model_totalloss_set, model_estop_set, outputDir=options.outputDir)
    plot_total_eff(model_set,model_eff_set,model_estop_set, outputDir=options.outputDir)
    plot_metric_vs_bitparam(model_set,prune_result_set,bit_params_set,base_acc_set,metric_text='ACC',outputDir=options.outputDir)
    plot_metric_vs_bitparam(model_set, prune_result_set, bit_params_set, base_roc_set, metric_text='ROC',outputDir=options.outputDir)
