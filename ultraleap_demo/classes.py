import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os


class HandPose():
    def __init__(self, model_path):
        self.model_path = os.path.abspath(model_path) if not os.path.isabs(model_path) else model_path
        separator = "\\" if "\\" in model_path else "/" if "/" in model_path else None
        self.model_type = model_path.split(separator)[-1].split("_")[0]
        self.model = self.make_model(self.model_type)


    def make_model(self, model_type):
        # Create the model
        if model_type == "filtereddhg":
            model = SiameseNetwork_filtered()
        elif model_type == "dhg":
            model = SiameseNetwork()
        else:
            raise ValueError("Invalid network complexity")
        
        # Load the model
        model.load_state_dict(torch.load(self.model_path))
        
        return model
    
    def get_similarity(self, angles1, angles2):
        angles1 = torch.from_numpy(angles1).float()
        angles2 = torch.from_numpy(angles2).float()

        angles1 = angles1.view(1, -1)
        angles2 = angles2.view(1, -1)

        self.model.eval()

        with torch.no_grad():
            score = self.model(angles1, angles2)

        return score.item()
    
    def get_similarities(self, angles):
        return np.array([self.get_similarity(angles[i], angles[i - 1]) for i in range(1, len(angles))])
            

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
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


class SiameseNetwork_filtered(nn.Module):
    def __init__(self):
        super(SiameseNetwork_filtered, self).__init__()
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