from preprocess import Preprocess
# from trainer import Trainer
from dhg import DHG
import numpy as np
import pickle
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DHGPATH = os.path.join(os.getcwd(), "DHGDATA")
FEATURESPATH = os.path.join(DHGPATH, "features")
PAIRSPATH = os.path.join(FEATURESPATH, "pairs")
LABELSPATH = os.path.join(FEATURESPATH, "labels")
MODELSPATH = os.path.join(DHGPATH, "models")

FILTEREDDHGPATH = os.path.join(os.getcwd(), "FILTEREDDHGDATA")
FILTEREDFEATURESPATH = os.path.join(FILTEREDDHGPATH, "features")
FILTEREDPAIRSPATH = os.path.join(FILTEREDFEATURESPATH, "pairs")
FILTEREDLABELSPATH = os.path.join(FILTEREDFEATURESPATH, "labels")
FILTEREDMODELSPATH = os.path.join(FILTEREDDHGPATH, "models")



class SiameseNetwork_dhg_1(nn.Module):
    def __init__(self):
        super(SiameseNetwork_dhg_1, self).__init__()
        # Define the activation function

        self.activation = nn.ReLU(inplace=True)
        layers = [nn.Linear(10, 48), nn.ReLU(inplace=True),
                nn.Linear(48, 10), nn.ReLU(inplace=True)]

        # Create the sequential model
        self.fc1 = nn.Sequential(*layers)
        self.fc2 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(inplace=True),
                                 nn.Linear(5, 1), nn.Sigmoid())

    def forward_once(self, x):
        output = self.fc1(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        L1_distance = torch.abs(output1 - output2)
        similarity_score = self.fc2(L1_distance) # If using BCEWithLogitsLoss no sigmoid needed
        return similarity_score
    
class SiameseNetwork_dhg_2(nn.Module):
    def __init__(self):
        super(SiameseNetwork_dhg_2, self).__init__()
        # Define the activation function

        self.activation = nn.ReLU(inplace=True)
        layers = [nn.Linear(10, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 10), nn.ReLU(inplace=True)]

        # Create the sequential model
        self.fc1 = nn.Sequential(*layers)
        self.fc2 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(inplace=True),
                                 nn.Linear(5, 1), nn.Sigmoid())

    def forward_once(self, x):
        output = self.fc1(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        L1_distance = torch.abs(output1 - output2)
        similarity_score = self.fc2(L1_distance) # If using BCEWithLogitsLoss no sigmoid needed
        return similarity_score


class SiameseNetwork_filtered(nn.Module):
    def __init__(self):
        super(SiameseNetwork_filtered, self).__init__()
        # Define the activation function

        self.activation = nn.ReLU(inplace=True)
        layers = [nn.Linear(10, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 10), nn.ReLU(inplace=True)]
        
        # Create the sequential model
        self.fc1 = nn.Sequential(*layers)
        self.fc2 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(inplace=True),
                                 nn.Linear(5, 1), nn.Sigmoid())

    def forward_once(self, x):
        output = self.fc1(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        L1_distance = torch.abs(output1 - output2)
        similarity_score = self.fc2(L1_distance) # If using BCEWithLogitsLoss no sigmoid needed
        return similarity_score
    
class SiameseNetwork_filtered_2(nn.Module):
    def __init__(self):
        super(SiameseNetwork_filtered_2, self).__init__()
        # Define the activation function

        self.activation = nn.ReLU(inplace=True)
        layers = [nn.Linear(10, 48), nn.ReLU(inplace=True),
                nn.Linear(48, 10), nn.ReLU(inplace=True)]

        # Create the sequential model
        self.fc1 = nn.Sequential(*layers)
        self.fc2 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(inplace=True),
                                 nn.Linear(5, 1), nn.Sigmoid())

    def forward_once(self, x):
        output = self.fc1(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        L1_distance = torch.abs(output1 - output2)
        similarity_score = self.fc2(L1_distance) # If using BCEWithLogitsLoss no sigmoid needed
        return similarity_score
    

def create_optimizer(model_parameters, learning_rate, weight_decay):
    optimizer = optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def make_model(model_type, learning_rate=0.001, weight_decay=0.0001):
    # Create the model
    if model_type == "dhg_48":
        model = SiameseNetwork_dhg_1()
    elif model_type == "filtered_32":
        model = SiameseNetwork_filtered()
    elif model_type == "filtered_48":
        model = SiameseNetwork_filtered_2()
    else:
        raise ValueError("Invalid network complexity")
    

    # Select optimizer
    optimizer = create_optimizer(model.parameters(), learning_rate, weight_decay)

    return model, optimizer

def configure_model(complexity, learning_rate=0.001, weight_decay=0.0001, filtered=False):
    # Create the model
    model, optimizer = make_model(complexity, learning_rate, weight_decay)

    criterion = nn.BCEWithLogitsLoss()
    label_range = [0, 1]

    # Define the preprocess object
    prep = None
    if filtered:
        prep = Preprocess(PAIRSPATH, LABELSPATH, train_size=0.8, random_state=42, batch_size=128, label_range=label_range)
    else:
        prep = Preprocess(FILTEREDPAIRSPATH, FILTEREDLABELSPATH, train_size=0.8, random_state=42, batch_size=128, label_range=label_range)
    return model, optimizer, criterion, prep