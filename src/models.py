import torch
import torch.nn as nn
from torch import optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GeneratorModel(nn.Module):
    def __init__(self, Natoms, noise_dimension, Nclasses, max_size, n1=50, n2=100, n3=200): 
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
    def __init__(self, Natoms, Nclasses, n1=200, n2=100, n3=50):
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

def training(discriminator, discriminator_optimizer, generator, generator_optimizer, loss, data_input, batch_size, noise_dim, N_classes, N_at):
    # Generate noise and move it the device
    noise = torch.randn(batch_size,noise_dim).to(device) 
    # Forward pass
    fake_labels = torch.randint(0,N_classes,(batch_size,)).to(device)
    generated_data = generator(noise, fake_labels) 
    
    true_data = data_input[0].view(batch_size, 3*N_at).to(device) 
    digit_labels = data_input[1].to(device) 
    true_labels = torch.ones(batch_size).to(device) 

    # Clear optimizer gradients        
    discriminator_optimizer.zero_grad()
    # Forward pass with true data as input
    discriminator_output_for_true_data = discriminator(true_data,digit_labels).view(batch_size) 
    # Compute Loss
    true_discriminator_loss = loss(discriminator_output_for_true_data, true_labels) 
    
    # Forward pass with generated data as input
    discriminator_output_for_generated_data = discriminator(generated_data.detach(), fake_labels).view(batch_size) 
    # Compute Loss
    generator_discriminator_loss = loss(discriminator_output_for_generated_data, torch.zeros(batch_size).to(device)) 
    # Average the loss
    discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2 
    # Backpropagate the losses for Discriminator model.
    discriminator_loss.backward()
    discriminator_optimizer.step()
    
    # Clear optimizer gradients
    generator_optimizer.zero_grad()        
    # It's a choice to generate the data again 
    generated_data = generator(noise, fake_labels) #.requires_grad_(False) 
    # Forward pass with the generated data
    discriminator_output_on_generated_data = discriminator(generated_data, fake_labels).view(batch_size) 
    # Compute loss: it must be the same of the discriminator, but reversed: the fake data must be passed as all ones, thus we use true_labels
    generator_loss = loss(discriminator_output_on_generated_data, true_labels) 
    # Backpropagate losses for Generator model.
    generator_loss.backward()
    generator_optimizer.step()

    return discriminator_loss.data.item(), generator_loss.data.item()

