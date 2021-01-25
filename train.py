import torch.nn as nn
import torch
import models
import jet_dataset
import matplotlib.pyplot as plt
from optparse import OptionParser
import torch.optim as optim
import math
import json
import numpy as np
from datetime import datetime
import os
import os.path as path

from training.train_funcs import train, val, test
from training.early_stopping import EarlyStopping
from tools.aiq import calc_AiQ
from tools.parse_yaml_config import parse_config
from tools.param_count import calc_BOPS


def create_model(parameterization, bits):
    try:
        dims = [parameterization['fc1s'], parameterization['fc2s'], parameterization['fc3s']]
    except:
        try:
            dims = [parameterization[0], parameterization[1], parameterization[2]]
        except Exception as e:
            print("Warning! Malformed node size array: {}".format(dims))
            print("Caught Exception: {}".format(e))

    prune_mask = {
        "fc1": torch.ones(dims[0], 16),
        "fc2": torch.ones(dims[1], dims[0]),
        "fc3": torch.ones(dims[2], dims[1]),
        "fc4": torch.ones(5, dims[2])}
    print("Creating model with the following dims:{}".format(dims))

    if bits is not 32:
        model = models.three_layer_model_bv_tunable(prune_mask, dims, precision=bits)
    else:
        model = models.three_layer_model_tunable(prune_mask, dims)  # 32b, non quantized model

    return model, prune_mask



if __name__ == "__main__": # If running, train a single model given some parameters
    parser = OptionParser()
    parser.add_option('-i','--input'   ,action='store',type='string',dest='inputFile'   ,default='', help='location of data to train off of')
    parser.add_option('-o','--output'   ,action='store',type='string',dest='outputDir'   ,default='train_simple/', help='output directory')
    parser.add_option('-t','--test'   ,action='store',type='string',dest='test'   ,default='', help='Location of test data set')
    parser.add_option('-l','--load', action='store', type='string', dest='modelLoad', default=None, help='Model to load instead of training new')
    parser.add_option('-c','--config'   ,action='store',type='string',dest='config'   ,default='configs/train_config_threelayer.yml', help='tree name')
    parser.add_option('-e','--epochs'   ,action='store',type='int', dest='epochs', default=250, help='number of epochs to train for')
    parser.add_option('-s', '--size', action='store', type='str', dest='size', default='64,32,32', help='Size of the model in format Hidden1,Hidden2,Hidden3')
    parser.add_option('-b', '--bits', action='store', type='int', dest='bits', default=32, help='bits of precision to quantize to')
    parser.add_option('-p', '--patience', action='store', type='int', dest='patience', default=10, help='Early Stopping patience in epochs')
    parser.add_option('-n', '--net_efficiency', action='store_true', dest='efficiency_calc', default=False, help='Enable Per-Epoch efficiency calculation (adds train time)')
    (options,args) = parser.parse_args()
    yamlConfig = parse_config(options.config)

    # create given output directory if it doesnt exist
    if not path.exists(options.outputDir):
        os.makedirs(options.outputDir, exist_ok=True)

    #current_model = models.three_layer_model() # Float Model
    model_size = [int(m) for m in options.size.split(',')]

    nbits = options.bits
    model, prune_mask = create_model(model_size, nbits) #Create quanized model of a given size and precision

    # Setup cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Using Device: {}".format(device))
    if use_cuda:
        print("cuda:0 device type: {}".format(torch.cuda.get_device_name(0)))

    # Set Batch size and split value
    batch_size = 1024
    train_split = 0.75

    # Setup and split dataset
    full_dataset = jet_dataset.ParticleJetDataset(options.inputFile, yamlConfig)
    test_dataset = jet_dataset.ParticleJetDataset(options.test, yamlConfig)
    train_size = int(train_split * len(full_dataset))  # 25% for Validation set, 75% for train set

    val_size = len(full_dataset) - train_size
    test_size = len(test_dataset)

    num_val_batches = math.ceil(val_size / batch_size)
    num_train_batches = math.ceil(train_size / batch_size)
    print("train_batches " + str(num_train_batches))
    print("val_batches " + str(num_val_batches))

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    print("train dataset size: " + str(len(train_dataset)))
    print("validation dataset size: " + str(len(val_dataset)))
    print("test dataset size: " + str(len(test_dataset)))

    # Setup dataloaders with our dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=10,
                                               pin_memory=True)  # FFS, have to use numworkers = 0 because apparently h5 objects can't be pickled, https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch/issues/69

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=10, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size,
                                              shuffle=False, num_workers=10, pin_memory=True)

    val_losses = []
    train_losses = []
    roc_auc_scores = []
    avg_precision_scores = []
    avg_train_losses = []
    avg_valid_losses = []
    accuracy_scores = []
    iter_eff = []

    early_stopping = EarlyStopping(patience=options.patience, verbose=True)

    model.update_masks(prune_mask)  # Make sure to update the masks within the model

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()

    L1_factor = 0.0001  # Default Keras L1 Loss
    estop = False
    epoch_counter = 0

    model.to(device)
    model.mask_to_device(device)

    if options.efficiency_calc and epoch_counter == 0:  # Get efficiency of un-initalized model
        aiq_dict, aiq_time = calc_AiQ(model, test_loader, True, device=device)
        epoch_eff = aiq_dict['net_efficiency']
        iter_eff.append(aiq_dict)
        print('[epoch 0] Model Efficiency: %.7f' % epoch_eff)
        for layer in aiq_dict["layer_metrics"]:
            print('[epoch 0]\t Layer %s Efficiency: %.7f' % (layer, aiq_dict['layer_metrics'][layer]['efficiency']))

    for epoch in range(options.epochs):  # loop over the dataset multiple times
        epoch_counter += 1
        # Train
        model, train_losses = train(model, optimizer, criterion, train_loader, L1_factor=L1_factor)

        # Validate
        val_losses, val_avg_precision_list, val_roc_auc_scores_list = val(model, criterion, val_loader,
                                                                          L1_factor=L1_factor)

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
            aiq_dict, aiq_time = calc_AiQ(model, test_loader, True, device=device)
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
            print('[epoch %d] Model Efficiency: %.7f' % (epoch + 1, epoch_eff))
            print('[epoch %d] aIQ Calc Time: %.7f seconds' % (epoch + 1, aiq_time))
            for layer in aiq_dict["layer_metrics"]:
                print('[epoch %d]\t Layer %s Efficiency: %.7f' % (
                epoch + 1, layer, aiq_dict['layer_metrics'][layer]['efficiency']))
        # Check if we need to early stop
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            estop = True
            epoch_counter -= options.patience
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


    # update our epoch counter to represent where the model actually stopped training
    epoch_counter -= ((len(avg_valid_losses)) - minposs)

    nbits = model.weight_precision if hasattr(model, 'weight_precision') else 32
    # Plot losses for this iter

    loss_ax.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    loss_ax.set_xlabel('epochs')
    loss_ax.set_ylabel('loss')
    loss_ax.grid(True)
    loss_ax.legend()
    filename = 'loss_plot_{}b_e{}_{}_.png'.format(nbits, epoch_counter, time)
    loss_ax.set_title('Loss from epoch 1 to {}, {}b model'.format(epoch_counter, nbits))
    loss_plt.savefig(path.join(options.outputDir, filename), bbox_inches='tight')
    loss_plt.show()
    plt.close(loss_plt)
    if options.efficiency_calc:
        # Plot & save eff for this iteration
        loss_plt = plt.figure()
        loss_ax = loss_plt.add_subplot()
        loss_ax.set_title('Net Eff. from epoch 0 to {}, {}b model'.format(epoch_counter, nbits))
        loss_ax.plot(range(0, len(iter_eff)), [z['net_efficiency'] for z in iter_eff],
                     label='Net Efficiency', color='green')

        # loss_ax.plot(range(1, len(iter_eff) + 1), [z["layer_metrics"][layer]['efficiency'] for z in iter_eff])
        loss_ax.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
        loss_ax.set_xlabel('epochs')
        loss_ax.set_ylabel('Net Efficiency')
        loss_ax.grid(True)
        loss_ax.legend()
        filename = 'eff_plot_{}b_e{}_{}_.png'.format(nbits, epoch_counter, time)
        loss_plt.savefig(path.join(options.outputDir, filename), bbox_inches='tight')
        loss_plt.show()
        plt.close(loss_plt)

    model_filename = "BO_{}b_best_{}.pth".format(nbits,time)
    torch.save(model.state_dict(), path.join(options.outputDir, model_filename))
    final_aiq, _ = calc_AiQ(model, test_loader, batnorm=True, device=device, full_results=True, testlabels=test_dataset.labels_list)

    model_totalloss_json_dict = {options.bits: [[avg_train_losses,avg_valid_losses], iter_eff, [minposs]]}

    filename = 'model_losses_{}_{}.json'.format(options.size,options.bits)
    with open(path.join(options.outputDir, filename), 'w') as fp:
        json.dump(model_totalloss_json_dict, fp)

    filename = 'model_AIQ_{}_{}.json'.format(options.size, options.bits)
    with open(path.join(options.outputDir, filename), 'w') as fp:
        json.dump({'{}b'.format(options.bits): {int(calc_BOPS(model, options.bits)): final_aiq, 'dims': str(model_size)}}, fp)

