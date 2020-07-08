import torch
import torchvision
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

import pandas as pd
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
from config import config 
from cnn_finetune import make_model
import os
from model.mynet import *

fold ='0'
# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 2
# 选择使用的网络
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net  = base_net_re()
    print(net)
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'

best_model = torch.load(config.best_models +config.model_name+os.sep+ str(fold) +os.sep+ 'lowest_loss.pth.tar')
net.load_state_dict(best_model["state_dict"])
net.eval()

# 数据处理，先缩放尺寸到（224*224），再变换数据类型为tensor,最后normalize
normalize = transforms.Normalize(
 mean=[0, 0, 0.],
 std=[1, 1, 1]
)
preprocess = transforms.Compose([
 transforms.Resize((64,64)),
 transforms.ToTensor(),
 normalize
])

#val_data = pd.read_csv('val.csv')
#for i in range(len(val_data['file_name']):

img_pil = Image.open('data/7.png')
img_pil = img_pil.convert("RGB")
img_tensor = preprocess(img_pil)
# 处理图片为Variable数据
img_variable = Variable(img_tensor.unsqueeze(0))
# 将图片输入网络得到回归结果
logit = net(img_variable)
print(logit)
y_pred = logit.cpu().detach().numpy().reshape(7,1)
if y_pred[0]>=0.5:
    y_pred[0]=1
else:
    y_pred[0]=0

if y_pred[1]>=0.7:
    y_pred[1]=1
else:
    y_pred[1]=0

y_test = np.array([[1.0,0.0,0.40852130325814534,0.29658605974395447,0.4848484848484848,0.1153619776339023,0.0049999999999999975]])
y_test = y_test.reshape(7,1)
print(y_test)
print(y_pred)
mse = mean_squared_error(y_test, y_pred)
print(mse)
r2 = r2_score(y_test, y_pred)
print(r2)
mae = mean_absolute_error(y_test, y_pred)
print(mae)


