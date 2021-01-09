import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets,transforms,models
import seaborn as sb
from collections import OrderedDict
import torchvision
from  classifier.classifierImageLoader import find_classes, make_dataset, default_loader
import torch.utils.data as data
import cv2
from PIL import Image
import os
os.environ['DISPLAY'] = ':0'

class SmokeData(data.Dataset):


    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mean = (0.479, 0.385, 0.352)
        self.std = (0.194, 0.171, 0.165)
        #获取类别和类别id
        classes, class_to_idx = find_classes(root_dir)
        samples = make_dataset(root_dir, class_to_idx)

        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(root_dir)
            raise RuntimeError(msg)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        print('------SmokeData[%s]',root_dir)
        print('classes[%s]'% self.classes)
        print('class_to_idx[%s]'%self.class_to_idx)
        self.count = 0
        #print('targets[%s]'%self.targets)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = default_loader(path)
        self.count = self.count+1
        #sample.save("./"+str(self.count)+'.jpg')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target



