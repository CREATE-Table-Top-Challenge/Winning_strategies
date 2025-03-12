import os
import numpy
from pathlib import Path
import argparse
import json
import gc
import math
import os
import sys
import time
import cv2
import pandas
import numpy
import torch
import scipy
import yaml
import random
import sklearn
import sklearn.metrics
from matplotlib import pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
import torch.nn.functional as F
from CNN import CNN
from DatasetGenerator import CNNDataset

def get_arguments():
    parser = argparse.ArgumentParser(description="Train a CNN model", add_help=False)
    # Data
    parser.add_argument(
        '--save_location',
        type=str,
        default='',
        help='Name of the directory where the models and results will be saved'
    )
    parser.add_argument(
        '--data_csv_file',
        type=str,
        default='',
        help='Path to the csv file containing locations for all data used in training'
    )
    ## CNN Parameters ##
    parser.add_argument(
        '--cnn_epochs',
        type=int,
        default=100,
        help='number of epochs for training cnn'
    )
    parser.add_argument(
        '--cnn_batch',
        type=int,
        default=16,
        help='batch size for training / validation of cnn'
    )
    parser.add_argument(
        '--balance_cnn',
        type=bool,
        default=True,
        help='Balance number of samples in each class for training'
    )
    parser.add_argument(
        '--augment_cnn',
        type=bool,
        default=True,
        help='Use augmentations when training CNN'
    )
    parser.add_argument(
        '--cnn_lr',
        type=float,
        default=1e-5,
        help='Learning rate for CNN optimizer'
    )
    parser.add_argument(
        '--cnn_features',
        type=int,
        default=128,
        help='Number of features in last layer of CNN before the final softmax'
    )

    ## General Parameters ##
    parser.add_argument(
        '--device',
        default='cuda',
        help='device to use for training / testing'
    )
    return parser

def loadData(datacsv):
    train_annotators = ['AN01', 'AN02', 'AN04', 'AN05', 'MS01', 'MS02', 'MS03', 'MS05']
    val_annotators = ['AN03', 'MS04']
    train_indexes = datacsv[datacsv['FileName'].str.contains('|'.join(train_annotators))].index
    val_indexes = datacsv[datacsv['FileName'].str.contains('|'.join(val_annotators))].index
    
    sets = {"Train": train_indexes, "Validation": val_indexes}
    datasets = []
    for learning_set in sets:
        print("Parsing {} data".format(learning_set.lower()))
        data = datacsv.iloc[sets[learning_set]].copy()
        data.index = [i for i in range(len(data.index))]
        data["Set"] = [learning_set for i in data.index]
        datasets.append(data)
    return datasets

def invert_class_mapping(class_mapping):
    inverted_mapping = {}
    for key in class_mapping:
        inverted_mapping[class_mapping[key]] = key
    print(inverted_mapping)
    return inverted_mapping

def writeConfig(foldDir, class_mapping, num_input_features, device):
    config = {"class_mapping":class_mapping,
              "num_features":num_input_features,
              "device":device}
    with open(os.path.join(foldDir,"config.yaml"),"w") as f:
        yaml.dump(config,f)

def writeResultsToFile(saveLocation,resultsDict,confmat):
    linesToWrite = []
    for key in resultsDict:
        if "Train" in key or "Val" in key:
            linesToWrite.append("\n{}: {}".format(key, resultsDict[key][-1]))
        else:
            linesToWrite.append("\n{}: {}".format(key,resultsDict[key]))
    linesToWrite.append("\n\nConfusion matrix")
    linesToWrite.append("\n" + str(confmat))
    with open(os.path.join(saveLocation, "trainingInfo_" + resultsDict["Model name"] + ".txt"), 'w') as f:
        f.writelines(linesToWrite)

def saveTrainingPlot(saveLocation,resultsDict,metric):
    fig = plt.figure()
    numEpochs =len(resultsDict["Train " + metric])
    plt.plot([x for x in range(numEpochs)], resultsDict["Train " + metric], 'bo', label='Training '+metric)
    plt.plot([x for x in range(numEpochs)], resultsDict["Val " + metric], 'b', label='Validation '+metric)
    plt.title(resultsDict["Model name"]+' Training and Validation ' + metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.savefig(os.path.join(saveLocation, resultsDict["Model name"]+'_'+metric + '.png'))
    plt.close(fig)

def trainResnet(foldDir, model,training_data,val_data,args,labelName="Overall Task"):
    gpu = torch.device(args.device)
    transforms = model.transforms
    resultsDict = {"Model name": "CNN",
                   "Num epochs": args.cnn_epochs,
                   "learning rate": args.cnn_lr,
                   "Train loss": [],
                   "Train accuracy": [],
                   "Val loss": [],
                   "Val accuracy": [],
                   "Final Val loss": None,
                   "Final Val accuracy": None}
    train_dataset = CNNDataset(training_data, labelName, transforms,balance=args.balance_cnn,augmentations = args.augment_cnn)
    val_dataset = CNNDataset(val_data, labelName, transforms,balance=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.cnn_batch,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.cnn_batch,
        num_workers=1,
        pin_memory=True,
    )
    classes = sorted(training_data[labelName].unique())
    lr = args.cnn_lr
    optimizer = optim.Adam(model.parameters(),lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    tool_loss = nn.BCEWithLogitsLoss()

    min_loss = 1e6
    num_epoch_without_improvement = 0
    for epoch in range(0, args.cnn_epochs):
        if num_epoch_without_improvement >= 4:
            print("Early stopping")
            break
        if num_epoch_without_improvement > 0 and num_epoch_without_improvement % 2 == 0:
            lr = adjust_learning_rate(lr, optimizer)#lr * 0.7
        model.train()
        loss_per_batch = []
        correct_instances = 0
        total_instances = 0
        start_time = last_logging = time.time()
        for step, (x, y) in enumerate(train_loader, start=epoch * len(train_loader)):
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)
            y_pred = model.forward(x)
            loss = loss_fn(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_per_batch.append(loss.item())
            y_pred = torch.softmax(y_pred, dim=1)
            categorical_y_pred = torch.argmax(y_pred, dim=1)
            categorical_y_true = y
            numCorrect = sum(categorical_y_pred == categorical_y_true).item()
            total_instances += len(y)
            correct_instances += numCorrect
            current_time = time.time()
            if current_time - last_logging > 30:
                stats = dict(
                    epoch=epoch,
                    step=step,
                    max_step=(epoch+1) * len(train_loader),
                    loss=loss.item(),
                    time=int(current_time - start_time),
                    estimated_time_remaining=int(((current_time - start_time)/(step-epoch*len(train_loader)))*((epoch+1)*len(train_loader)-step)),
                    lr=lr,
                )
                print(json.dumps(stats))
                last_logging = current_time
        state = dict(
            epoch=epoch + 1,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        train_accuracy = correct_instances / total_instances
        model.eval()
        val_loss_per_batch = []
        correct_instances = 0
        total_instances = 0
        start_time = last_logging = time.time()
        with torch.no_grad():
            for val_step, (x, y) in enumerate(val_loader, start=epoch * len(val_loader)):
                x = x.cuda(gpu, non_blocking=True)
                y = y.cuda(gpu, non_blocking=True)
                y_pred = model.forward(x)
                val_loss = loss_fn(y_pred, y)
                y_pred = torch.softmax(y_pred, dim=1)

                val_loss_per_batch.append(val_loss.item())
                categorical_y_pred = torch.argmax(y_pred, dim=1)
                categorical_y_true = y
                numCorrect = sum(categorical_y_pred == categorical_y_true).item()
                correct_instances += numCorrect
                total_instances += len(y)
                current_time = time.time()
                if current_time - last_logging > 30:
                    stats = dict(
                        epoch=epoch,
                        mode="Val",
                        step=val_step,
                        loss=val_loss.item(),
                        time=int(current_time - start_time),
                        estimated_time_remaining=int(
                            ((current_time - start_time) / (val_step - epoch * len(val_loader))) * ((epoch + 1) * len(val_loader) - val_step)),
                        lr=lr,
                    )
                    print(json.dumps(stats))
                    last_logging = current_time
        val_accuracy = correct_instances / total_instances
        train_loss = sum(loss_per_batch) / len(loss_per_batch)
        val_loss = sum(val_loss_per_batch) / len(val_loss_per_batch)
        resultsDict["Train loss"].append(train_loss)
        resultsDict["Train accuracy"].append(train_accuracy)
        resultsDict["Val loss"].append(val_loss)
        resultsDict["Val accuracy"].append(val_accuracy)
        if val_loss < min_loss:
            print("Val loss decreased from {} to {}. saving model.".format(min_loss, val_loss))
            torch.save(state, os.path.join(foldDir, "resnet.pth"))
            min_loss = val_loss
            num_epoch_without_improvement = 0
        else:
            num_epoch_without_improvement += 1
        print("Epoch: {} - Train loss: {}, Train accuracy: {}, Val loss: {}, Val accuracy: {}".format(epoch, train_loss,
                                                                                                      train_accuracy,
                                                                                                      val_loss,
                                                                                                      val_accuracy))
        if args.balance_cnn:
            train_dataset.balanceDataByVideo()

    ckpt = torch.load(os.path.join(foldDir, "resnet.pth"), map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    val_loss_per_batch = []
    correct_instances = 0
    total_instances = 0
    pred_labels = numpy.array([])
    true_labels = numpy.array([])
    with torch.no_grad():
        for val_step, (x, y) in enumerate(val_loader):
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)
            y_pred = model.forward(x)
            val_loss = loss_fn(y_pred, y)
            y_pred = torch.softmax(y_pred, dim=1)
            val_loss_per_batch.append(val_loss.item())
            categorical_y_pred = torch.argmax(y_pred, dim=1)
            categorical_y_true = y
            numCorrect = sum(categorical_y_pred == categorical_y_true).item()
            correct_instances += numCorrect
            total_instances += len(x)
            pred_labels = numpy.concatenate([pred_labels, categorical_y_pred.cpu().numpy()], axis=None)
            true_labels = numpy.concatenate([true_labels, categorical_y_true.cpu().numpy()], axis=None)
    val_accuracy = correct_instances / total_instances
    val_loss = sum(val_loss_per_batch) / len(val_loss_per_batch)
    confMat = sklearn.metrics.confusion_matrix(true_labels, pred_labels)
    print("Final Testing Stats: loss: {}, accuracy: {}".format(val_loss, val_accuracy))
    print(confMat)
    resultsDict["Final Val loss"] = [val_loss]
    resultsDict["Final Val accuracy"] = [val_accuracy]
    writeResultsToFile(foldDir, resultsDict, confMat)
    saveTrainingPlot(foldDir, resultsDict, "loss")
    saveTrainingPlot(foldDir, resultsDict, "accuracy")

def main(args):
    if args.save_location == "":
        print("No save location specified. Please set flag --save_location")
    elif args.data_csv_file == "":
        print("No dataset specified. Please set flag --data_csv_file")
    else:
        torch.backends.cudnn.benchmark = True
        print(args)
        gpu = torch.device(args.device)
        labelName = "Overall Task"
        dataCSVFile = pandas.read_csv(args.data_csv_file)
        num_classes = len(dataCSVFile[labelName].unique())
        foldDir = args.save_location
        if not os.path.exists(foldDir):
            os.mkdir(foldDir)
        args.save_location = foldDir
        network = CNN()

        train_data,val_data = loadData(dataCSVFile) #args.validation_percentage
        class_counts = train_data[labelName].value_counts()
        classes = sorted(dataCSVFile[labelName].unique())
        print(class_counts)
        class_mapping = dict(zip([i for i in range(len(dataCSVFile[labelName].unique()))],
                                 sorted(dataCSVFile[labelName].unique())))

        num_input_features = args.cnn_features
        writeConfig(foldDir, class_mapping, num_input_features, args.device)
        
        resnetModel = network.createCNNModel(num_input_features, num_classes).cuda(gpu)
        if not os.path.exists(os.path.join(foldDir, "resnet.pth")):
            trainResnet(foldDir, resnetModel,train_data,val_data,args,labelName=labelName)


def adjust_learning_rate(lr, optimizer):
    lr = lr*0.7
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

if __name__ == "__main__":
    parser = argparse.ArgumentParser('CNN training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
