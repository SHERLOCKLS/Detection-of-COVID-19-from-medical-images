from torch.utils.data import Dataset
from torchvision import transforms as T 
from config import config
from PIL import Image 
from itertools import chain 
from glob import glob
from tqdm import tqdm
import random 
import numpy as np 
import pandas as pd 
import os 
import cv2
import torch 
#from .autoaugment import ImageNetPolicy

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#1.set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

#2.define dataset
'''
                    T.Normalize(mean = [0.485,0.456,0.406],
                                std = [0.229,0.224,0.225])])
'''
class ChaojieDataset(Dataset):
    def __init__(self,label_list,transforms=None,train=True,test=False):
        self.test = test 
        self.train = train 
        imgs = []
        if self.test:
            for index,row in label_list.iterrows():
                imgs.append((row["filename"]))
            self.imgs = imgs 
        else:
            for index,row in label_list.iterrows():
                imgs.append((row["filename"],row["label"]))
            self.imgs = imgs
        if transforms is None:
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize((config.img_weight,config.img_height)),
                    T.ToTensor(),
                    #T.Normalize(mean = [0.485,0.456,0.406],
                    #            std = [0.229,0.224,0.225])])
                    T.Normalize(mean = [0, 0, 0],
                                std = [1, 1, 1])])

            else:
                self.transforms  = T.Compose([
                    T.Resize((config.img_weight,config.img_height)),
#                    Cutout(),
#                    T.RandomHorizontalFlip(),
#                    T.ColorJitter(0.4, 0.4, 0.4),
#                    T.RandomRotation(30),
                    T.RandomHorizontalFlip(),
#                    ImageNetPolicy(),
#                    T.RandomVerticalFlip(),
#                    T.RandomAffine(45),
                    T.ToTensor(),
                    #T.Normalize(mean = [0.485,0.456,0.406],
                    #            std = [0.229,0.224,0.225])])
                    T.Normalize(mean = [0, 0, 0],
                                std = [1, 1, 1])])
        else:
            self.transforms = transforms
    def __getitem__(self,index):
        if self.test:
            filename = self.imgs[index]
            print(filename)

            img = Image.open(filename)
            img = img.convert("RGB")
            img = self.transforms(img)
            return img,filename


        else:
            filename,label = self.imgs[index] 
            #print(filename)
            img = Image.open(filename)
            img = img.convert("RGB")
            img = self.transforms(img)
            return img,label
    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), \
           label

def get_files(root,mode):
    #for test
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename":files})
        return files
    elif mode != "test": 
        #for train and val
        df = pd.read_csv('train_lab-x.csv')
        
        all_data_path,labels = [],[]
        for i in range(len(df['name'])):
            all_data_path.append(root+df['name'][i])
            labels.append(int(df['label'][i]))

        #image_folders = list(map(lambda x:root+x,os.listdir(root)))
        #jpg_image_1 = list(map(lambda x:glob(x+"/*.png"),image_folders))
        #jpg_image_2 = list(map(lambda x:glob(x+"/*.PNG"),image_folders))
        #all_images = list(chain.from_iterable(jpg_image_1 + jpg_image_2))
        print("loading train dataset")
        #for file in tqdm(all_images):
        #    all_data_path.append(file)
        #    labels.append(int(file.split("/")[-2]))
        all_files = pd.DataFrame({"filename":all_data_path,"label":labels})
        return all_files
    else:
        print("check the mode please!")
    
