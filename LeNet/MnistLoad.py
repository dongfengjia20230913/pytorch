import os
import struct
import numpy as np

import matplotlib.pyplot as plt

    
def load_mnist_train(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'train-labels.idx1-ubyte')
    images_path = os.path.join(path,'train-images.idx3-ubyte')
    with open(labels_path, 'rb') as lbpath:
        #'>II' 大端模式(big-endian),
        magic, n = struct.unpack('>II',lbpath.read(8))#读取8个字节
        labels = np.fromfile(lbpath,dtype=np.uint8)
        print(magic,n,labels)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 28*28)
        print(magic,num,rows,cols)
        print(images.shape)

    return images, labels

def show(images,labels):
    fig, ax = plt.subplots(
        nrows=2,
        ncols=5,
        sharex=True,
        sharey=True, )
 
    ax = ax.flatten()
    for i in range(10):
         #根据i获取labels中与i相同的数据的位置，然后通过这个位置id获取对应的images图片数据
        img = images[labels == i][0].reshape(28, 28)
 
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
 
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    
    
images, labels  = load_mnist_train('../datas/Mnist')
show(images,labels)
print([labels==5])