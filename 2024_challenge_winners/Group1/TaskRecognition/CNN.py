import numpy
import os
import cv2
import torch
from torch import nn, optim
from torchvision import models
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.functional as F
import yaml
import copy

class CNN:
    def __init__(self):
        self.cnn_model = None
        self.task_class_mapping = None

    def loadModel(self, modelFolder, modelName=None):
        self.loadConfig(modelFolder)
        self.loadCNNModel(modelFolder)

    def loadConfig(self,modelFolder):
        with open(os.path.join(modelFolder,"config.yaml"),"r") as f:
            config = yaml.safe_load(f)
        self.task_class_mapping = config["class_mapping"]
        self.num_features = config["num_features"]
        self.num_classes = len([key for key in self.task_class_mapping])
        self.device = config["device"]

    def loadCNNModel(self,modelFolder):
        self.cnn_model = ResNet_FeatureExtractor(self.num_features, self.num_classes)
        res_ckpt = torch.load(os.path.join(modelFolder, "resnet.pth"), map_location="cpu")
        self.cnn_model.load_state_dict(res_ckpt["model"], strict=True)
        self.transforms = self.cnn_model.transforms
        try:
            self.cnn_model.cuda(self.device)
        except AttributeError:
            pass
        return self.cnn_model

    def predict(self,image):
        self.cnn_model.eval()
        with torch.no_grad():
            img_tensor = self.transforms(image.resize((224,224)))
            image = torch.from_numpy(numpy.array([img_tensor])).cuda(self.device)
            pred = self.cnn_model.forward(image)
            pred = torch.softmax(pred, dim=1)
            class_num = torch.argmax(pred, dim=1)
            class_num = class_num.cpu().numpy()
            taskPrediction = pred.cpu().numpy().tolist()
            class_num = class_num[0]
            networkOutput = str(self.task_class_mapping[class_num]) + str(taskPrediction)
            print(networkOutput)
            return networkOutput

    def createCNNModel(self,num_input_features,num_classes):
        self.cnn_model = ResNet_FeatureExtractor(num_input_features, num_classes)
        self.num_classes = num_classes
        self.num_features = num_input_features
        return self.cnn_model

class ResNet_FeatureExtractor(nn.Module):
    def __init__(self,num_output_features,num_classes,multitask=False,num_tools=0,return_head=True):
        super(ResNet_FeatureExtractor,self).__init__()
        weights = models.ResNet50_Weights.DEFAULT
        #weights = models.Inception_V3_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.resnet = models.resnet50(weights=weights)
        #self.resnet = models.inception_v3(weights=weights)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.linear1 = nn.Linear(num_features,num_output_features)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.output_head = nn.Linear(num_output_features,num_classes)
        if multitask:
            self.tool_head = nn.Linear(num_output_features,num_tools)
        #self.softmax = nn.Softmax(dim=1)
        self.return_head = return_head
        self.multitask = multitask

    def forward(self,x):
        x = self.resnet(x)
        #if not torch.is_tensor(x):
            #x = x[0]
        x = self.linear1(x)
        x = self.sig(x)
        #x = self.relu(x)
        if self.return_head:
            if self.multitask:
                task = self.output_head(x)
                tool = self.tool_head(x)
                return task,tool
            else:
                x = self.output_head(x)
            #x = self.softmax(x)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()
    def forward(self,x):
        return x