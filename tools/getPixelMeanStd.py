import os
import cv2
from numpy import *

img_dir='./datas/smoke_clas/train' #训练图片目录
img_list=os.listdir(img_dir)
img_size=64
sum_r=0
sum_g=0
sum_b=0

stdevs_r = 0
stdevs_g = 0
stdevs_b = 0
count=0

imgLists = []

def collect(frompath):
 
    files = os.listdir(frompath)
    for fi in files:
        fi_d = os.path.join(frompath,fi)
        if os.path.isdir(fi_d):
            collect(fi_d)
        elif fi_d.endswith('jpg'):
            imagePath = os.path.join(fi_d)
            imagename = os.path.basename(imagePath)
            global imgLists
            imgLists.append(imagePath)

collect(img_dir)
for img_name in imgLists:
    img_path=img_name
    print(img_path)
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(img_size,img_size))
    sum_r=sum_r+img[:,:,0].mean()
    sum_g=sum_g+img[:,:,1].mean()
    sum_b=sum_b+img[:,:,2].mean()
    
    stdevs_r=stdevs_r+img[:,:,0].std()
    stdevs_g=stdevs_g+img[:,:,1].std()
    stdevs_b=stdevs_b+img[:,:,2].std()

    count=count+1

count = count*255
sum_r=sum_r/count
sum_g=sum_g/count
sum_b=sum_b/count

stdevs_r=stdevs_r/count
stdevs_g=stdevs_g/count
stdevs_b=stdevs_b/count


img_mean=[sum_r,sum_g,sum_b]
img_std=[stdevs_r,stdevs_g,stdevs_b]
print ('img_mean:',img_mean)
print ('img_std:',img_std)