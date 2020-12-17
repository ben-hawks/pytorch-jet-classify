import torch
from tools import TensorEfficiency
import time as time_lib
import brevitas.nn as qnn
import numpy as np

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

def calc_AiQ(aiq_model, test_loader, batnorm = True, device='cpu'):
    """ Calculate efficiency of network using TensorEfficiency """
    # Time the execution
    start_time = time_lib.time()
    aiq_model.cpu()
    aiq_model.mask_to_device('cpu')
    aiq_model.eval()
    hooklist = []
    # Set up the data
    ensemble = {}

    # Initialize arrays for storing microstates
    if batnorm:
        microstates = {name: np.ndarray([]) for name, module in aiq_model.named_modules() if
                       ((isinstance(module, torch.nn.Linear) or isinstance(module, qnn.QuantLinear)) and name == 'fc4') \
                       or (isinstance(module, torch.nn.BatchNorm1d))}
        microstates_count = {name: 0 for name, module in aiq_model.named_modules() if
                             ((isinstance(module, torch.nn.Linear) or isinstance(module,qnn.QuantLinear)) and name == 'fc4') \
                             or (isinstance(module, torch.nn.BatchNorm1d))}
    else:
        microstates = {name: np.ndarray([]) for name, module in aiq_model.named_modules() if
                       isinstance(module, torch.nn.Linear) or isinstance(module, qnn.QuantLinear)}
        microstates_count = {name: 0 for name, module in aiq_model.named_modules() if
                             isinstance(module, torch.nn.Linear) or isinstance(module, qnn.QuantLinear)}

    activation_outputs = SaveOutput()  # Our forward hook class, stores the outputs of each layer it's registered to

    # register a forward hook to get and store the activation at each Linear layer while running
    layer_list = []
    for name, module in aiq_model.named_modules():
        if batnorm:
            if ((isinstance(module, torch.nn.Linear) or isinstance(module, qnn.QuantLinear)) and name == 'fc4') \
              or (isinstance(module, torch.nn.BatchNorm1d)):  # Record @ BN output except last layer (since last has no BN)
                hooklist.append(module.register_forward_hook(activation_outputs))
                layer_list.append(name)  # Probably a better way to do this, but it works,
        else:
            if (isinstance(module, torch.nn.Linear) or isinstance(module,qnn.QuantLinear)):  # We only care about linear layers except the last
                hooklist.append(module.register_forward_hook(activation_outputs))
                layer_list.append(name)  # Probably a better way to do this, but it works,
    # Process data using torch dataloader, in this case we
    for i, data in enumerate(test_loader, 0):
        activation_outputs.clear()
        local_batch, local_labels = data

        # Run through our test batch and get inference results
        with torch.no_grad():
            local_batch, local_labels = local_batch.to('cpu'), local_labels.to('cpu')
            outputs = aiq_model(local_batch.float())

            # Calculate microstates for this run
            for name, x in zip(layer_list, activation_outputs.outputs):
                # print("---- AIQ Calc ----")
                # print("Act list: " + name + str(x))
                x = x.numpy()
                # Initialize the layer in the ensemble if it doesn't exist
                if name not in ensemble.keys():
                    ensemble[name] = {}

                # Initialize an array for holding layer states if it has not already been initialized
                sort_count_freq = 1  # How often (iterations) we sort/count states
                if microstates[name].size == 1:
                    microstates[name] = np.ndarray((sort_count_freq * np.prod(x.shape[0:-1]), x.shape[-1]), dtype=bool,
                                                   order='F')

                # Store the layer states
                new_count = microstates_count[name] + np.prod(x.shape[0:-1])
                microstates[name][
                microstates_count[name]:microstates_count[name] + np.prod(x.shape[0:-1]), :] = np.reshape(x > 0,(-1, x.shape[-1]), order='F')
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
            layer_metrics[layer] = {key: value for key, value in
                                    zip(metrics, TensorEfficiency.layer_efficiency(states))}
        for hook in hooklist:
            hook.remove() #remove our output recording hooks from the network

        # Calculate network efficiency and aIQ, with beta=2
        net_efficiency = TensorEfficiency.network_efficiency([m['efficiency'] for m in layer_metrics.values()])
        #print('AiQ Calc Execution time: {}'.format(time_lib.time() - start_time))
        # Return AiQ along with our metrics
        aiq_model.to(device)
        aiq_model.mask_to_device(device)
        return {'net_efficiency': net_efficiency, 'layer_metrics': layer_metrics}, (time_lib.time() - start_time)
