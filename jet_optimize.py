import torch.nn as nn
import torch
import numpy as np
import models
import jet_dataset
from optparse import OptionParser
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from iter_prune import l1_regularizer, parse_config
import torch.optim as optim
import math
from tools.pytorchtools import EarlyStopping
from datetime import datetime
import os
import os.path as path
import ax
import matplotlib.pyplot as plt
import json
#from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train

tested_sizes = []


def train(model, optimizer, loss, train_loader, L1_factor=0.0001):
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
        reg_loss = l1_regularizer(model, lambda_l1=L1_factor)
        total_loss = criterion_loss + reg_loss
        total_loss.backward()
        optimizer.step()
        step_loss = total_loss.item()
        train_losses.append(step_loss)
    return model, train_losses


def val(model, loss, val_loader, L1_factor=0.0001):
    val_roc_auc_scores_list = []
    val_avg_precision_list = []
    val_losses = []
    model.to(device)
    model.mask_to_device(device)
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
            val_roc_auc_scores_list.append(roc_auc_score(local_labels.numpy(), outputs.numpy()))
            val_avg_precision_list.append(average_precision_score(local_labels.numpy(), outputs.numpy()))
            val_losses.append(val_loss)
    return val_losses, val_avg_precision_list, val_roc_auc_scores_list


def create_model(parameterization):
    dims = [parameterization['fc1s'], parameterization['fc2s'], parameterization['fc3s']]
    tested_sizes.append(dims)
    prune_masks = {
        "fc1": torch.ones(dims[0], 16),
        "fc2": torch.ones(dims[1], dims[0]),
        "fc3": torch.ones(dims[2], dims[1]),
        "fc4": torch.ones(5, dims[2])}
    print("Creating model with the following dims:{}".format(dims))
    model = models.three_layer_model_bv_tunable(prune_masks, dims, precision=options.bits)

    return model

def run_train(model,train_loader,val_loader):

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()

    L1_factor = 0.0001  # Default Keras L1 Loss

    early_stopping = EarlyStopping(patience=options.patience, verbose=True)

    valid_losses = []
    train_losses_epoch = []
    estop = 0

    for epoch in range(options.epochs):  # loop over the dataset multiple times
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

        # Print epoch statistics
        print('[epoch %d] train batch loss: %.7f' % (epoch + 1, train_loss))
        print('[epoch %d] val batch loss: %.7f' % (epoch + 1, valid_loss))
        print('[epoch %d] val ROC AUC Score: %.7f' % (epoch + 1, val_roc_auc_score))
        print('[epoch %d] val Avg Precision Score: %.7f' % (epoch + 1, val_avg_precision))

        valid_losses.append(valid_loss.tolist())
        train_losses_epoch.append(train_loss.tolist())

        # Check if we need to early stop
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            estop = epoch-10
            break
        else:
            estop = epoch

    # Load last/best checkpoint model saved via earlystopping
    model.load_state_dict(torch.load('checkpoint.pt'))
    # Time for filenames
    now = datetime.now()
    time = now.strftime("%d-%m-%Y_%H-%M-%S")
    if not path.exists('{}b'.format(os.path.join(options.outputDir, str(options.bits)))):
        os.makedirs('{}b'.format(os.path.join(options.outputDir,  str(options.bits))))
    filename = "{}b/BO_{}b_JetModel_{}-{}-{}_{}.pth".format(os.path.join(options.outputDir,  str(options.bits)), options.bits, model.dims[0], model.dims[1], model.dims[2],time)
    print("saving model to {}".format(filename))
    torch.save(model.state_dict(),filename)

    # Plot & save losses for this iteration
    loss_plt = plt.figure()
    loss_ax = loss_plt.add_subplot()
    loss_plt_filename = "{}b/BO_{}b_losses_{}-{}-{}_{}.png".format(os.path.join(options.outputDir,  str(options.bits)),
                                                                   options.bits,
                                                                   model.dims[0], model.dims[1],model.dims[2],
                                                                   time)
    #print("Val Losses: {}".format(valid_losses))
    #print("Train Losses: {}".format(train_losses_epoch))

    loss_ax.plot(range(1, len(train_losses_epoch) + 1), train_losses_epoch, label='Training Loss')
    loss_ax.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')

    nbits = model.weight_precision if hasattr(model, 'weight_precision') else 32
    # Plot losses for this iter

    loss_ax.axvline(estop+1, linestyle='--', color='r', label='Early Stopping Checkpoint')
    loss_ax.set_xlabel('epochs')
    loss_ax.set_ylabel('loss')
    loss_ax.grid(True)
    loss_ax.legend()
    loss_ax.set_title('Loss from epoch {} to {}, {}b model'.format(0, estop, nbits))
    loss_plt.savefig(loss_plt_filename, bbox_inches='tight')
    loss_plt.show()
    plt.close(loss_plt)

    model_loss_dict.update({'{}-{}-{}'.format(model.dims[0], model.dims[1],model.dims[2]): [train_losses_epoch,valid_losses]})

    return model


def test(model, test_loader, L1_factor=0.0001):
    #predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    #lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    test_loss_value_list = []
    test_criterion = nn.BCELoss()
    model.to(device)
    model.mask_to_device(device)
    with torch.no_grad():  # Evaulate pruned model performance
        for i, data in enumerate(test_loader):
            model.eval()
            local_batch, local_labels = data
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch.float())
            #_, preds = torch.max(outputs, 1)
            #predlist = torch.cat([predlist, preds.view(-1).cpu()])
            #lbllist = torch.cat([lbllist, torch.max(local_labels, 1)[1].view(-1).cpu()])
            criterion_loss = test_criterion(outputs, local_labels.float())
            reg_loss = 0  # l1_regularizer(model, lambda_l1=L1_factor)
            test_loss = criterion_loss + reg_loss
        #test_loss_value_list.append(accuracy_score(lbllist.numpy(), predlist.numpy()))
        #accuracy = np.average(accuracy_score_value_list)
        test_loss_value_list.append(test_loss)
        try:
            loss = np.average(test_loss_value_list)
        except:
            loss = torch.mean(torch.stack(test_loss_value_list)).cpu().numpy()

    # add test loss to our loss dict to make plotting stuff easier, we calc other perf metrics later
    model_loss_dict['{}-{}-{}'.format(model.dims[0], model.dims[1],model.dims[2])].append(loss.tolist())
    print("Loss on test set for model: {}".format(loss))
    return loss #accuracy

def create_train_eval(parameterization):
    model = create_model(parameterization)

    trained_model = run_train(model,train_loader,validate_loader)

    eval_loss = test(trained_model,test_loader)

    print("eval return dtype: {}".format(type(eval_loss)))

    return float(eval_loss)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-i','--input'   ,action='store',type='string',dest='inputFile'   ,default='', help='location of data to train off of')
    parser.add_option('-o','--output'   ,action='store',type='string',dest='outputDir'   ,default='train_simple/', help='output directory')
    parser.add_option('-t','--test'   ,action='store',type='string',dest='test'   ,default='', help='Location of test data set')
    parser.add_option('-l','--load', action='store', type='string', dest='modelLoad', default=None, help='Model to load instead of training new')
    parser.add_option('-c','--config'   ,action='store',type='string',dest='config'   ,default='configs/train_config_threelayer.yml', help='tree name')
    parser.add_option('-e','--epochs'   ,action='store',type='int', dest='epochs', default=100, help='number of epochs to train for')
    parser.add_option('-p', '--patience', action='store', type='int', dest='patience', default=10,help='Early Stopping patience in epochs')
    parser.add_option('-b', '--bits', action='store', type='int', dest='bits', default=8, help='Bits of precision to quantize model to')
    parser.add_option('-n', '--trials', action='store', type='int', dest='trials', default=20, help='Number of trials to perform')
    (options,args) = parser.parse_args()
    yamlConfig = parse_config(options.config)

    # Setup cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Using Device: {}".format(device))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
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

    validate_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=10, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size,
                                              shuffle=False, num_workers=10, pin_memory=True)

    #Loss export stuff:
    model_loss_dict = {}

    best_parameters, values, experiment, model = ax.optimize(
        parameters=[
            {"name": "fc1s", "type": "range", "bounds": [8, 64]},
            {"name": "fc2s", "type": "range", "bounds": [4, 32]},
            {"name": "fc3s", "type": "range", "bounds": [4, 32]}
        ],
        evaluation_function=create_train_eval,
        minimize=True,  # minimize since objective is loss
        total_trials=options.trials,  # default 20
        arms_per_trial=1,  # default 1
        objective_name='BCE_loss',
    )

    filename = 'BO_model_losses_{}b.json'.format(options.bits)
    with open(os.path.join(options.outputDir, filename), 'w') as fp:
        json.dump(model_loss_dict, fp)
    print(str(model_loss_dict))

    print("~*~*~*~*~*~*~*~RESULTS~*~*~*~*~*~*~*~")
    print("Best Params: {}".format(best_parameters))
    means, covariances = values
    print("Means: {}".format(means))
    print("Covariances: {}".format(covariances))

    # Time for filenames
    now = datetime.now()
    time = now.strftime("%d-%m-%Y_%H-%M-%S")
    result_filename = "{}BO_Results_Summary-{}b_{}.txt".format(options.outputDir,options.bits,time)
    results_file = open(result_filename,"a")
    for sizes in tested_sizes:
        results_file.write("Tested size: {}\n".format(sizes))
    results_file.write("Best Params: {}\n".format(best_parameters))
    results_file.write("Means: {}\n".format(means))
    results_file.write("Covariances: {}\n".format(covariances))
    print("Results saved to {}".format(result_filename))