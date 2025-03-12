import torch
import numpy
import random
import os
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import math
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler,MinMaxScaler

class CNNDataset(Dataset):
    def __init__(self,datacsv,labelName,transforms,balance=True,img_size = 224,augmentations=False):
        self.datacsv = datacsv.copy()
        self.transforms = transforms
        self.balanced = balance
        self.data = self.datacsv.index.copy()
        self.labelName = labelName
        self.labels = sorted(self.datacsv[self.labelName].unique())
        self.balanceDataByVideo()
        self.img_size = img_size
        self.augmentations = augmentations
        self.currentIndexes = dict(zip([i for i in range(len(self.labels))],[0 for i in range(len(self.labels))]))

    def __len__(self):
        if self.balanced:
            minCount = math.inf
            for label in self.labels:
                if len(self.sample_mapping[label]) < minCount:
                    minCount = len(self.sample_mapping[label])
            return round(minCount*len(self.labels)) #
        else:
            return len(self.data)

    def convertCategoricalToOneHot(self,label):
        one_hot_label = numpy.zeros((len(self.labels)))
        idx = self.labels.index(label)
        one_hot_label[idx] = 1
        return idx #one_hot_label.astype(float)

    def balanceDataByVideo(self,num_samples = 100):
        print("Resampling data")
        self.sample_mapping = {}
        videos = sorted(self.datacsv["Folder"].unique())
        for vid in videos:
            for label in self.labels:
                entries = self.datacsv.loc[(self.datacsv[self.labelName] == label) & (self.datacsv["Folder"]==vid)]
                if not entries.empty:
                    sample_indexes = random.choices(entries.index.copy(),k=num_samples)
                    if label in self.sample_mapping:
                        self.sample_mapping[label] += shuffle(sample_indexes)
                    else:
                        self.sample_mapping[label] = shuffle(sample_indexes)

    def splitDataByClass(self):
        self.sample_mapping = {}
        for label in self.labels:
            entries = self.datacsv.loc[self.datacsv[self.labelName]==label]
            self.sample_mapping[label] = shuffle(entries.index.copy())

    def rotateImage(self,image,angle = -1):
        if angle < 0:
            angle = random.randint(1, 359)
        center = tuple(numpy.array(image.shape[1::-1])/2)
        rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
        rotImage = cv2.warpAffine(image,rot_mat,image.shape[1::-1],flags=cv2.INTER_LINEAR)
        return rotImage

    def getBalancedSample(self):
        idx = random.randint(0, len(self.labels) - 1)
        sample_idx = self.currentIndexes[idx]
        label = self.labels[idx]
        sample = self.sample_mapping[label][sample_idx]
        imgFilePath = os.path.join(self.datacsv["Folder"][sample],self.datacsv["FileName"][sample])
        img_tensor = self.transforms(Image.open(imgFilePath).resize((self.img_size,self.img_size)))
        preprocessing = random.randint(0,10)
        if self.augmentations:
            img = img_tensor.cpu().numpy()
            if preprocessing == 0:
                # flip along y axis
                img = cv2.flip(img, 1)
            elif preprocessing == 1:
                # flip along x axis
                img = cv2.flip(img, 0)
            elif preprocessing == 2:
                # rotate
                angle = random.randint(1, 359)
                img = self.rotateImage(img, angle)
            img_tensor = torch.from_numpy(img)
        self.currentIndexes[idx] = (self.currentIndexes[idx] + 1) % (len(self.sample_mapping[label]))
        if self.currentIndexes[idx] == 0:
            for key in self.sample_mapping:
                self.sample_mapping[key] = shuffle(self.sample_mapping[key])
        label = self.convertCategoricalToOneHot(label)
        label_tensor = torch.tensor(label)
        return img_tensor,label_tensor

    def getNextSample(self,idx):
        sample = self.data[idx]
        imgFilePath = os.path.join(self.datacsv["Folder"][sample], self.datacsv["FileName"][sample])
        img_tensor = self.transforms(Image.open(imgFilePath).resize((self.img_size,self.img_size)))
        label = self.datacsv[self.labelName][sample]
        label = self.convertCategoricalToOneHot(label)
        label_tensor = torch.tensor(label)
        return img_tensor, label_tensor

    def __getitem__(self,idx):
        if self.balanced == True:
            sequence_tensor,label_tensor = self.getBalancedSample()
        else:
            sequence_tensor, label_tensor = self.getNextSample(idx)
        return (sequence_tensor,label_tensor)