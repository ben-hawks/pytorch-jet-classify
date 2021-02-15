# Import misc packages
import os


# Import torch stuff
import torch
import torch.onnx


# Import our own code
import models
import jet_dataset



if __name__ == "__main__":
    model_path = "../model_files/32b_70pruned_0rand.pth"
    model = models.three_layer_model_batnorm(bn_affine=True, bn_stats=True)
    model.load_state_dict(torch.load(os.path.join(model_path), map_location='cpu'))

    exportname="32b_70Pruned_FullModel"

    dummy_input = torch.randn(1,16)
    torch.onnx.export(model, dummy_input, exportname+".onnx", verbose=True)#, input_names=input_names)#, output_names=output_names)
    torch.save(model,exportname+".pth")