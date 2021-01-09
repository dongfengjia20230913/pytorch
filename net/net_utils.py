import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn as nn


def save_model(model, optimizer, epoch, net_type, data_type, dummy_input ):
    out_dir = os.path.join('./models', net_type + '_' + data_type)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    path = os.path.join(out_dir, data_type+'_%d.pth'%(epoch))
    print('save net:\n    [%s]'%path)
    torch.save(model.state_dict(),path)
    model_all_path = os.path.join(out_dir, data_type+'_%d_all.pth'%(epoch))
    torch.save(model,model_all_path)
    onnx_path = os.path.join(out_dir, data_type+'_%d.onnx'%(epoch))
    torch.onnx.export(model, dummy_input, onnx_path)


def load_model(model,path):
    model.load_state_dict(torch.load(path))
    return model