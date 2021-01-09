import os
from torchvision.utils import make_grid
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def get_layer(layers, cur_index, layer_count):
    cur_layer = layers[cur_index]
    next_layer = "NULL"
    next_layer_index = cur_index+1
    if 'Conv2d' in str(cur_layer) or 'MaxPool2d' in str(cur_layer):
        if next_layer_index < layer_count:
            for j in range(next_layer_index, layer_count):
                if 'Conv2d' in str(layers[j]) or 'MaxPool2d' in str(layers[j]):
                    next_layer = layers[j]
                    next_layer_index = j
                    break
    else:
        cur_layer = "NULL"
    return cur_layer, cur_index, next_layer, next_layer_index

def is_conv2d(layer):
    return 'Conv2d' in str(layer)

def is_maxPool2d(layer):
    return 'MaxPool2d' in str(layer) 

def is_linear(layer):
    return 'Linear' in str(layer) 

def save_img_linear(tensor, name):
    #替换深度和batch_size所在的纬度值
    tensor = tensor.permute((1, 0))
    #print('output permute:',tensor.shape)
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.cpu().data.numpy() * 255.).astype(np.uint8)#将0~1之间的像素值，转化成0~255
    Image.fromarray(im).save(name + '.jpg')

def save_all_featuremap_gray(tensor, name):
    tensor = tensor.permute((1, 0, 2, 3))#将[1, 6, 28, 28]转化成[6, 1, 28, 28]
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.cpu().data.numpy() * 255.).astype(np.uint8)#将0~1之间的像素值，转化成0~255
    Image.fromarray(im).save(name)

def save_all_featuremap_colors(tensor, name):
    tensor = tensor.permute((1, 0, 2, 3))#将[1, 6, 28, 28]转化成[6, 1, 28, 28]
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2)
    im = (im[0].cpu().data.numpy() * 255.).astype(np.uint8)#将0~1之间的像素值，转化成0~255
    plt.imshow(im)
    plt.savefig(name)


def write_con2d_pool_layer_output(layer_out_image, output, is_convd, cur_index):
    if not os.path.exists(output):
        os.makedirs(output)
    if is_convd:
        temp_img = layer_out_image
        save_all_featuremap_gray(  layer_out_image,output + "/" + str(cur_index)+"_conv2d_gray_out.jpg")
        save_all_featuremap_colors(layer_out_image,output + "/" + str(cur_index)+"_conv2d_color_out.jpg")
        temp_img = temp_img.cpu().detach().numpy()[0]
        temp_img = np.sum(temp_img, 0)
        plt.imshow(temp_img)
        plt.savefig(output + "/" + str(cur_index)+"_conv2d_one_out.jpg")
    else :
        temp_img = layer_out_image
        save_all_featuremap_gray(   layer_out_image, output + "/" + str(cur_index)+"_pool_gray_out.jpg")
        save_all_featuremap_colors( layer_out_image, output + "/" + str(cur_index)+"_pool_color_out.jpg")
        temp_img = temp_img.cpu().detach().numpy()[0]
        temp_img = np.max(temp_img, 0)
        plt.imshow(temp_img)
        plt.savefig(output + "/" + str(cur_index)+"_pool_one_out.jpg")


def write_linear_layer_output(layer_out_image, output, is_linear, cur_index):
    if not os.path.exists(output):
        os.makedirs(output)
    save_name = output + "/" + "classifier_"+str(cur_index)+".jpg"
    if is_linear:
        save_name = output + "/" + "classifier_linear_"+str(cur_index)+".jpg"
    print('saving linear:',save_name)
    np.savetxt(save_name+'.txt', layer_out_image.cpu().data.numpy() ,fmt="%s")
    save_img_linear(layer_out_image ,save_name)

def write_con2d_pool_layers_output(image, layers, output):
    if not os.path.exists(output):
        os.makedirs(output)
    layer_count=len(layers)
    for i in range(layer_count):
        current_layer = layers[i]
        current_layer, cur_index, nextLayer, next_index = get_layer(layers, i, layer_count)
        if current_layer != "NULL":
            layer_out = current_layer(image)
            write_con2d_pool_layer_output(layer_out, output, is_conv2d(nextLayer), cur_index)
            image = layer_out;
    return image


def write_linear_layers_output(image, layers, output):
    if not os.path.exists(output):
        os.makedirs(output)
    layer_count=len(layers)
    for i in range(layer_count):
        current_layer = layers[i]
        layer_out = current_layer(image)
        print('layer_out.shpae:',layer_out.shape)
        write_linear_layer_output(layer_out, output, is_linear(current_layer), i)
        image = layer_out
    return image
 