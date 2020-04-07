import torch
import torchvision.models as models
import numpy as np
from LSUV import LSUVinit
import sys
import os
sys.path.insert(0, '/home/ubuntu/dev/opencv-3.1/build/lib')
import cv2
from torch.autograd import Variable
images_to_process = []
for img_fname in os.listdir('imgs'):
    img = cv2.imread('imgs/' + img_fname)
    print (img.shape)
    if img is not None:
        images_to_process.append(np.transpose(cv2.resize(img, (224,224)), (2,0,1) ))
        
data = np.array(images_to_process).astype(np.float32)
data = torch.from_numpy(data)
alexnet = models.densenet121(pretrained=False)
alexnet = LSUVinit(alexnet,data, needed_std = 1.0, std_tol = 0.1, max_attempts = 10, needed_mean = 0., do_orthonorm = False)
