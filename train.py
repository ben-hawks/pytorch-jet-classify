import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import models
import jet_dataset
import h5py
import matplotlib.pyplot as plt
from optparse import OptionParser
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, average_precision_score,precision_recall_curve
import torch.optim as optim
import yaml
from torchsummaryX import summary
import math
import seaborn as sn

def get_features(options, yamlConfig):
    # To use one data file:
    h5File = h5py.File(options.inputFile)
    treeArray = h5File[options.tree][()]

    print(treeArray.shape)
    print(treeArray.dtype.names)

    # List of features to use
    features = yamlConfig['Inputs']

    # List of labels to use
    labels = yamlConfig['Labels']

    # Convert to dataframe
    features_labels_df = pd.DataFrame(treeArray, columns=list(set(features + labels)))
    features_labels_df = features_labels_df.drop_duplicates()

    features_df = features_labels_df[features]
    labels_df = features_labels_df[labels]

    if 'Conv' in yamlConfig['InputType']:
        labels_df = labels_df.drop_duplicates()

    # Convert to numpy array
    features_val = features_df.values
    labels_val = labels_df.values

    if 'j_index' in features:
        features_val = features_val[:, :-1]  # drop the j_index feature
    if 'j_index' in labels:
        labels_val = labels_val[:, :-1]  # drop the j_index label
        print(labels_val.shape)

    if yamlConfig['InputType'] == 'Conv1D':
        features_2dval = np.zeros((len(labels_df), yamlConfig['MaxParticles'], len(features) - 1))
        for i in range(0, len(labels_df)):
            features_df_i = features_df[features_df['j_index'] == labels_df['j_index'].iloc[i]]
            index_values = features_df_i.index.values
            # features_val_i = features_val[index_values[0]:index_values[-1]+1,:-1] # drop the last feature j_index
            features_val_i = features_val[np.array(index_values), :]
            nParticles = len(features_val_i)
            # print("before", features_val_i[:,0])
            features_val_i = features_val_i[
                features_val_i[:, 0].argsort()[::-1]]  # sort descending by first value (ptrel, usually)
            # print("after", features_val_i[:,0])
            if nParticles > yamlConfig['MaxParticles']:
                features_val_i = features_val_i[0:yamlConfig['MaxParticles'], :]
            else:
                features_val_i = np.concatenate(
                    [features_val_i, np.zeros((yamlConfig['MaxParticles'] - nParticles, len(features) - 1))])
            features_2dval[i, :, :] = features_val_i

        features_val = features_2dval

    elif yamlConfig['InputType'] == 'Conv2D':
        features_2dval = np.zeros((len(labels_df), yamlConfig['BinsX'], yamlConfig['BinsY'], 1))
        for i in range(0, len(labels_df)):
            features_df_i = features_df[features_df['j_index'] == labels_df['j_index'].iloc[i]]
            index_values = features_df_i.index.values

            xbins = np.linspace(yamlConfig['MinX'], yamlConfig['MaxX'], yamlConfig['BinsX'] + 1)
            ybins = np.linspace(yamlConfig['MinY'], yamlConfig['MaxY'], yamlConfig['BinsY'] + 1)

            x = features_df_i[features[0]]
            y = features_df_i[features[1]]
            w = features_df_i[features[2]]

            hist, xedges, yedges = np.histogram2d(x, y, weights=w, bins=(xbins, ybins))

            for ix in range(0, yamlConfig['BinsX']):
                for iy in range(0, yamlConfig['BinsY']):
                    features_2dval[i, ix, iy, 0] = hist[ix, iy]
        features_val = features_2dval

    X_train_val, X_test, y_train_val, y_test = train_test_split(features_val, labels_val, test_size=0.2,
                                                                random_state=42)

    # Normalize inputs
    if yamlConfig['NormalizeInputs'] and yamlConfig['InputType'] != 'Conv1D' and yamlConfig['InputType'] != 'Conv2D':
        scaler = preprocessing.StandardScaler().fit(X_train_val)
        X_train_val = scaler.transform(X_train_val)
        X_test = scaler.transform(X_test)

    # Normalize inputs
    if yamlConfig['NormalizeInputs'] and yamlConfig['InputType'] != 'Conv1D' and yamlConfig['InputType'] != 'Conv2D' and \
            yamlConfig['KerasLoss'] == 'squared_hinge':
        scaler = preprocessing.MinMaxScaler().fit(X_train_val)
        X_train_val = scaler.transform(X_train_val)
        X_test = scaler.transform(X_test)
        y_train_val = y_train_val * 2 - 1
        y_test = y_test * 2 - 1

    # Normalize conv inputs
    if yamlConfig['NormalizeInputs'] and yamlConfig['InputType'] == 'Conv1D':
        reshape_X_train_val = X_train_val.reshape(X_train_val.shape[0] * X_train_val.shape[1], X_train_val.shape[2])
        scaler = preprocessing.StandardScaler().fit(reshape_X_train_val)
        for p in range(X_train_val.shape[1]):
            X_train_val[:, p, :] = scaler.transform(X_train_val[:, p, :])
            X_test[:, p, :] = scaler.transform(X_test[:, p, :])

    if 'j_index' in labels:
        labels = labels[:-1]

    return X_train_val, X_test, y_train_val, y_test, labels

## Config module
def parse_config(config_file) :

    print("Loading configuration from", config_file)
    config = open(config_file, 'r')
    return yaml.load(config)



if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-i','--input'   ,action='store',type='string',dest='inputFile'   ,default='', help='input file')
    parser.add_option('-o','--output'   ,action='store',type='string',dest='outputDir'   ,default='train_simple/', help='output directory')
    parser.add_option('-t','--tree'   ,action='store',type='string',dest='tree'   ,default='t_allpar_new', help='tree name')
    parser.add_option('-c','--config'   ,action='store',type='string',dest='config'   ,default='configs/train_config_threelayer.yml', help='tree name')
    parser.add_option('-e','--epochs'   ,action='store',type='int', dest='epochs', default=100, help='number of epochs to train for')
    (options,args) = parser.parse_args()
    print(options.config)
    yamlConfig = parse_config(options.config)

    current_model = models.three_layer_model()
    summary(current_model,torch.zeros(16))
    current_model.double() #compains about getting doubles when expecting floats without this. Might be a problem with quantization, but dtypes *should* be handled better then



    # CUDA for PyTorch
    use_cuda = False #torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    #X_train_val, X_test, y_train_val, y_test, labels = get_features(options,yamlConfig)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(current_model.parameters(), lr=0.0001)

    batch_size = 1000
    full_dataset = jet_dataset.ParticleJetDataset(options,yamlConfig)
    train_size = int(0.75 * len(full_dataset)) #25% for Validation set, 75% for train set
    val_size = len(full_dataset) - train_size
    num_val_batches = math.ceil(val_size/batch_size)
    num_train_batches = math.ceil(train_size/batch_size)
    print("train_batches " + str(num_train_batches))
    print("val_batches " + str(num_val_batches))
    train_dataset, val_dataset =  torch.utils.data.random_split(full_dataset,[train_size,val_size])#Figure out data loading

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=0) #FFS, have to use numworkers = 0 because apparently h5 objects can't be pickled, https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch/issues/69

    val_loader =   torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    val_losses = []
    train_losses = []
    roc_auc_scores = []
    avg_precision_scores = []
    accuracy_scores = []
    L1_alpha = yamlConfig['L1RegR'] if 'L1RegR' in yamlConfig else 0.01  # Keras default value if not specified
    for epoch in range(options.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        # Training
        #for local_batch, local_labels in train_loader:
        for i, data in enumerate(train_loader, 0):
            local_batch, local_labels = data
            current_model.train()
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = current_model(local_batch)
            criterion_loss = criterion(outputs, torch.max(local_labels, 1)[1]) #via https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216/2
            #l1 reg on weights
            L1_reg = torch.tensor(0., requires_grad=True)
            for name, param in current_model.named_parameters():
                if 'weight' in name:
                    L1_reg = L1_reg + torch.norm(param, 1)
            total_loss = criterion_loss + (L1_alpha * L1_reg)
            total_loss.backward()
            optimizer.step()
            step_loss = total_loss.item()
            #train_losses.append(step_loss)
            if i == num_train_batches-1: #print every 8 batches for less console spam
                print('[epoch %d, batch: %1d] train batch loss: %.7f' % (epoch + 1, i + 1, step_loss))
                train_losses.append(step_loss)
            if epoch % 100 == 0:
                print(outputs[0])
            # Validation
        with torch.set_grad_enabled(False):
            current_model.eval()
            for i, data in enumerate(val_loader, 0):
                local_batch, local_labels = data
                outputs = current_model(local_batch)
                L1_reg = torch.tensor(0., requires_grad=True)
                for name, param in current_model.named_parameters():
                    if 'weight' in name:
                        L1_reg = L1_reg + torch.norm(param, 1)
                val_loss = criterion(outputs, torch.max(local_labels, 1)[1])  + (L1_alpha * L1_reg)
                #print(local_labels.numpy())
                #print(outputs.numpy())
                val_roc_auc_score = roc_auc_score(local_labels.numpy(), outputs.numpy())
                val_avg_precision = average_precision_score(local_labels.numpy(), outputs.numpy())
                #roc_auc_scores.append(val_roc_auc_score)
                if i == num_val_batches-1:  # print every 8 batches for less console spam
                    print('[epoch %d, val batch: %1d] val batch loss: %.7f' % (epoch + 1, i + 1, val_loss))
                    print('[epoch %d, val batch: %1d] val ROC AUC Score: %.7f' % (epoch + 1, i + 1, val_roc_auc_score))
                    print('[epoch %d, val batch: %1d] val Avg Precision Score: %.7f' % (epoch + 1, i + 1, val_avg_precision))
                    val_losses.append(val_loss)
                    roc_auc_scores.append(val_roc_auc_score)
                    avg_precision_scores.append(val_avg_precision)

    print("ROC AUC Table size: " + str(len(roc_auc_scores)))
    plt.plot(train_losses,color='r',linestyle='solid', alpha=0.3)
    plt.plot(val_losses, color='g',linestyle='dashed')
    plt.legend(['Train Loss', 'Val Loss'], loc='upper left')
    plt.ylabel("Batch Loss (perk 1k samples)")
    plt.xlabel("Epoch")
    plt.show()
    plt.plot(roc_auc_scores,color='r',linestyle='solid', alpha=0.3)
    plt.ylabel("ROC AUC")
    plt.xlabel("Epoch")
    plt.show()
    plt.plot(avg_precision_scores,color='r',linestyle='solid', alpha=0.3)
    plt.ylabel("Avg Precision")
    plt.xlabel("Epoch")
    plt.show()

    # Initialize the prediction and label lists(tensors)
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    outlist = torch.zeros(0, dtype=torch.long, device='cpu')
    prob_labels = torch.zeros(0, dtype=torch.long, device='cpu')
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            current_model.eval()
            local_batch, local_labels = data
            outputs = current_model(local_batch)
            _, preds = torch.max(outputs, 1)
            print(preds)

            # Append batch prediction results
            #outlist = torch.cat([outlist, outputs.cpu().type(torch.DoubleTensor)])
            #prob_labels = torch.cat([prob_labels, local_labels.cpu().type(torch.DoubleTensor)])
            predlist = torch.cat([predlist, preds.view(-1).cpu()])
            lbllist = torch.cat([lbllist, torch.max(local_labels, 1)[1].view(-1).cpu()])
            print(lbllist)
            print(predlist)
            #val_roc_curve = roc_curve(local_labels.numpy(), outputs.numpy())

    # Confusion matrix
    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())

    df_cm = pd.DataFrame(conf_mat, index=[i for i in full_dataset.labels_list],
                         columns=[i for i in full_dataset.labels_list])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True,fmt='g')

    plt.show()


    print(conf_mat)

    print('Finished Training')