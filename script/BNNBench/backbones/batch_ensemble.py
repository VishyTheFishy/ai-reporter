import torch
from torchvision.models.densenet import densenet121
from torchvision.models.resnet import resnet50
from BNNBench.backbones.vgg import VGG
from BNNBench.backbones.unet import define_G
from BNNBench.backbones.batch_conv2d import BatchConv2d, BatchConvTrans2d

def modify_model(model, ensemble_size):
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            cout, cin, k, k = layer.weight.shape
            batch_ensemble_layer = BatchConv2d(cin, cout, k, layer.padding, layer.stride, ensemble_size)
            batch_ensemble_layer.weight.data.copy_(layer.weight.data)
            if layer.bias is not None:
                batch_ensemble_layer.bias.data.copy_(layer.bias.data)
            path = name.split('.')
            tgt_layer = model 
            for idx, val in enumerate(path):
                if idx == len(path)-1:
                    setattr(tgt_layer, str(val), batch_ensemble_layer)
                else:
                    tgt_layer = getattr(tgt_layer, str(val))
                    
        if isinstance(layer, torch.nn.ConvTranspose2d):
            cin, cout, k, k = layer.weight.shape
            batch_ensemble_layer = BatchConvTrans2d(cin, cout, k, padding=layer.padding, stride=layer.stride, ensemble_size=ensemble_size)
            #import pdb
            #pdb.set_trace()
            batch_ensemble_layer.weight.data.copy_(layer.weight.data)
            if layer.bias is not None:
                batch_ensemble_layer.bias.data.copy_(layer.bias.data)
            path = name.split('.')
            tgt_layer = model 
            for idx, val in enumerate(path):
                if idx == len(path)-1:
                    setattr(tgt_layer, str(val), batch_ensemble_layer)
                else:
                    tgt_layer = getattr(tgt_layer, str(val))

def batch_vgg(vgg_name, ensemble_size, *args, **kwargs):
    model = VGG(vgg_name, *args, **kwargs)
    modify_model(model, ensemble_size)
    return model

def batch_densenet121(ensemble_size, *args, **kwargs):
    model = densenet121(*args, **kwargs)
    modify_model(model, ensemble_size)
    return model

def batch_resnet50(ensemble_size, **kwargs):
    model = resnet50(**kwargs)
    modify_model(model, ensemble_size)
    return model

def batch_unet(ensemble_size, *args, **kwargs):
    G = define_G(*args, **kwargs)
    modify_model(G, ensemble_size)
    return G.cuda()