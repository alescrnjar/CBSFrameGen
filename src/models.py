import torch
import torch.nn as nn
from torch import optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GeneratorModel(nn.Module):
    #def __init__(self,Natoms,Nclasses,max_size):
    def __init__(self,Natoms,noise_dimension,Nclasses,max_size, n1=50, n2=100, n3=200): 
        super(GeneratorModel, self).__init__() 
        input_dim = noise_dimension + Nclasses 
        output_dim = 3*Natoms
        n1 = 50
        n2 = 100
        n3 = 200
        self.max_size = max_size
        self.label_embedding = nn.Embedding(Nclasses, Nclasses) 
        self.hidden_layer1 = nn.Sequential(nn.Linear(input_dim, n1), nn.LeakyReLU(0.2)) 
        self.hidden_layer2 = nn.Sequential(nn.Linear(n1, n2), nn.LeakyReLU(0.2)) 
        self.hidden_layer3 = nn.Sequential(nn.Linear(n2, n3), nn.LeakyReLU(0.2)) 
        self.output_layer = nn.Sequential(nn.Linear(n3, output_dim), nn.Tanh()) 
    
    def forward(self, x, labels): 
        c = self.label_embedding(labels) 
        x = torch.cat([x,c],1) 
        output = self.hidden_layer1(x) 
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        output = self.output_layer(output) 
        output = self.max_size*output
        return output.to(device)

class DiscriminatorModel(nn.Module):  
    def __init__(self,Natoms,Nclasses, n1=200, n2=100, n3=50):
        super(DiscriminatorModel, self).__init__() 
        input_dim = 3*Natoms + Nclasses 
        output_dim = 1
        self.label_embedding = nn.Embedding(Nclasses, Nclasses) 
        self.hidden_layer1 = nn.Sequential(nn.Linear(input_dim, n1), nn.LeakyReLU(0.2), nn.Dropout(0.3)) 
        self.hidden_layer2 = nn.Sequential(nn.Linear(n1, n2), nn.LeakyReLU(0.2), nn.Dropout(0.3)) 
        self.hidden_layer3 = nn.Sequential(nn.Linear(n2, n3), nn.LeakyReLU(0.2), nn.Dropout(0.3)) 
        self.output_layer = nn.Sequential(nn.Linear(n3, output_dim), nn.Sigmoid())
        
    def forward(self, x, labels): 
        c = self.label_embedding(labels)
        x = torch.cat([x, c], 1)
        output = self.hidden_layer1(x)
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        output = self.output_layer(output)
        return output.to(device)
