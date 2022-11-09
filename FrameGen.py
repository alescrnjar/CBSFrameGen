import torch
import torch.nn as nn
from torch import optim as optim
#
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array 
from numpy.linalg import norm
import MDAnalysis.analysis.rms
from MDAnalysis.analysis import align
from MDAnalysisData import datasets
#
import numpy as np
import os
import matplotlib.pyplot as plt
#
import parmed
#
from tensorboardX import SummaryWriter  
#
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--biosystem', default='PROTEIN', type=str) # Necessary for atom selections (see below)
parser.add_argument('--train_mode', default=True, type=bool) # Set to False for Test mode.
parser.add_argument('--load_model', default=False, type=bool)
parser.add_argument('--desired_format', default='inpcrd', type=str)
parser.add_argument('--N_classes', default=2, type=int) # Number of classes
parser.add_argument('--desired_labels', default=[0,1], type=list) # Classes to be considered for output
parser.add_argument('--noise_dim', default=100, type=int) # Dimension for gaussian noise to feed to the generator
parser.add_argument('--n_structures', default=10, type=int) # Number of structures to be generated for every class (only in Test mode)
parser.add_argument('--epoch_freq', default=10, type=int) # Output will be generated every this many epochs
parser.add_argument('--log_freq', default=100, type=int) # Verbose output will be printed every this many epochs
parser.add_argument('--learning_rate', default=0.0002, type=float) # Learning rate for Adam optimizer
parser.add_argument('--dist_cut', default=10.0, type=float) # distance cut-off for class defintion
parser.add_argument('--input_directory', default='./example_input/' , type=str) 
parser.add_argument('--output_directory', default='./example_output/', type=str) 
parser.add_argument('--n_epochs', default=1000, type=int) # Number of epochs for training
parser.add_argument('--batch_size', default=100, type=int) # Batch size for training
args = parser.parse_args()

prmf = args.input_directory+'peptide.prmtop' # Parameter and topology file
#trajfs = [args.input_directory+'all_conformations.mdcrd'] # Trajectory files list
trajfs = ['/home/acrnjar/Desktop/TEMP/Peptides_gen/all_conformations.mdcrd'] # Trajectory files list

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

print("Output will be written in:",args.output_directory)
if not os.path.exists(args.output_directory): os.system('mkdir '+args.output_directory)

print("Cuda is available:",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.biosystem=='PROTEIN':
    backbone='name CA C N'
elif args.biosystem=='DNA':
    backbone='name P'

print("train_mode:",args.train_mode)
print("load_model:",args.load_model)
if args.train_mode:
    outmode='train'
else:
    outmode='eval'

model_g_file=args.output_directory+'model_generator.pth'
model_d_file=args.output_directory+'model_discriminator.pth'

# Remove previous output if necessary
if args.train_mode:
    os.system('rm '+args.output_directory+'gen*'+args.desired_format+' 2> /dev/null')
    if not args.load_model:
        os.system('rm '+args.output_directory+'out*'+args.desired_format+' 2> /dev/null')
        
last_epoch=0
if args.train_mode and args.load_model:
    prefixed = [filename for filename in os.listdir(args.output_directory) if filename.startswith("out_train_label1_epoch")]
    past_epochs=[]
    for wfile in prefixed:
        past_epochs.append(int(wfile.replace('out_train_label1_epoch','').replace('.'+desired_format,'')))
    last_epoch=max(past_epochs)
    print("Last epoch found:",last_epoch)

### ### ### FUNCTIONS

def write_inpcrd(inp_tensor,outname='out.inpcrd'):
    """
    Given input coordinates tensor, writes .inpcrd file.
    """
    inp_tensor=list(np.array(inp_tensor))
    outf=open(outname,'w')
    outf.write('default_name\n'+str(len(inp_tensor))+'\n')
    for at in range(len(inp_tensor)):
        outf.write(" {:11.7f} {:11.7f} {:11.7f}".format(float(inp_tensor[at][0]),float(inp_tensor[at][1]),float(inp_tensor[at][2])))
        if (at%2!=0):
            outf.write('\n')
    outf.close()

def max_size(prm_top_file,trajectory_file,selection='all',factor=1.0):
    """
    Maximum value that any coordinate can have. This requires the system to be centered on the origin.
    """
    universe=mda.Universe(prm_top_file,trajectory_file)
    all_maxm=[]
    for ts in universe.trajectory[::1]:
        pos=universe.select_atoms(selection).positions
        maxm=0.0
        for at in pos:
            for coord in at:
                if (maxm<np.sqrt(coord*coord)): maxm=np.sqrt(coord*coord)
        all_maxm.append(maxm)
    # A factor can be multiplied in order to allow some fluctuations on the protein surface
    return(factor*max(all_maxm))

def bonds_deviation(prm_top_file,inpcrd_file):
    """
    Root mean square deviation of bonds with respect to their equilibrium value (defined by the force field).
    """
    myparams=parmed.amber.readparm.AmberParm(prm_top_file,xyz=inpcrd_file)
    bonds=parmed.tools.actions.printBonds(myparams,'!(:WAT,Na+,Cl-)') 
    dev2s=[]
    for line in str(bonds).split('\n'):
        if ('Atom' not in line and len(line.split())!=0):
            Req=float(line.split()[10])
            Distance=float(line.split()[8])
            dev2s.append((Req-Distance)**2)
    return np.sqrt(np.mean(dev2s))

def angles_deviation(prm_top_file,inpcrd_file):
    """
    Root mean square deviation of angles with respect to their equilibrium value (defined by the force field). 
    """
    myparams=parmed.amber.readparm.AmberParm(prm_top_file,xyz=inpcrd_file)
    angles=parmed.tools.actions.printAngles(myparams,'!(:WAT,Na+,Cl-)') 
    dev2s=[]
    for line in str(angles).split('\n'):
        if ('Atom' not in line and len(line.split())!=0):
            Theta_eq=float(line.split()[13])
            Angle=float(line.split()[14])
            difference=abs(Angle-Theta_eq)
            while difference>180.: difference-=180.
            dev2s.append(difference)
    return np.sqrt(np.mean(dev2s))

def check_label_condition(prm_top_file,inpcrd_file):
    """
    Returns the observable responsible for the class definition.
    """
    u=mda.Universe(prm_top_file,inpcrd_file)
    pos_dstz_at1=u.select_atoms('resid 1 and name CA').center_of_mass()
    pos_dstz_at2=u.select_atoms('resid 6 and name CA').center_of_mass()
    return norm(pos_dstz_at1-pos_dstz_at2) 
    
def generate_training_data(prm_top_file,traj_file,frame_i,frame_f):
    """
    Generate training dataset.
    """
    u=mda.Universe(prm_top_file,traj_file)
    input_dats=[]
    count_0=0
    count_1=0
    at_list=[]
    listed=False

    ref_u=u
    ref_u.trajectory[0]
    ref_pos=ref_u.select_atoms(backbone).positions - ref_u.atoms.center_of_mass() #backbone works with both proteins and DNA
    
    for ts in u.trajectory[frame_i:frame_f:1]:

        # Align the current frame to the first one
        prot_pos=u.select_atoms(backbone).positions - u.atoms.center_of_mass()
        R_matrix, R_rmsd = align.rotation_matrix(prot_pos,ref_pos)
        u.atoms.translate(-u.select_atoms(backbone).center_of_mass())
        u.atoms.rotate(R_matrix)
        u.atoms.translate(ref_u.select_atoms(backbone).center_of_mass())

        sel=u.select_atoms('all')

        # Make atom list with atom names and coordinates
        if not listed:
            for atx in range(len(sel.atoms.ids)):
                at_sel=u.select_atoms('bynum '+str(atx+1))
                at_list.append([])
                at_list[-1].append(at_sel.residues.resids[0])
                at_list[-1].append(at_sel.residues.resnames[0])
                at_list[-1].append(at_sel.atoms.names[0])
                at_list[-1].append(at_sel.atoms.ids[0])
                listed=True

        # Define observable for labeling data
        pos_dstz_at1=u.select_atoms('resid 1 and name CA').center_of_mass()
        pos_dstz_at2=u.select_atoms('resid 6 and name CA').center_of_mass()
        dist_dstz=norm(pos_dstz_at1-pos_dstz_at2)

        # Assign label 
        if (dist_dstz<args.dist_cut): 
            lab_val=1
            count_1+=1
        else:
            lab_val=0
            count_0+=1
        
        input_dats.append((torch.tensor(sel.positions),lab_val)) 
        if (ts.frame==0):
            write_inpcrd(sel.positions,outname=args.output_directory+'initial.inpcrd')
    input_dataset=input_dats
    print("{} frames with label 0, {} frames with label 1.".format(count_0,count_1))
    return input_dataset,at_list

### ### ### MODELS

class GeneratorModel(nn.Module):
    #def __init__(self,Natoms,Nclasses,max_size):
    def __init__(self,Natoms,noise_dimension,Nclasses,max_size): 
        super(GeneratorModel, self).__init__() 
        input_dim = noise_dimension + Nclasses 
        output_dim = 3*Natoms
        n1=50
        n2=100
        n3=200
        self.max_size=max_size
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
    def __init__(self,Natoms,Nclasses):
        super(DiscriminatorModel, self).__init__() 
        input_dim = 3*Natoms + Nclasses 
        output_dim = 1
        n1=200
        n2=100
        n3=50 
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

### ### ### MAIN

def main():

    univ=mda.Universe(prmf,trajfs)
    nframes=len(univ.trajectory)

    dataset,atoms_list=generate_training_data(prmf,trajfs,0,nframes-(nframes%args.batch_size))

    N_at=len(univ.select_atoms('all'))
    print("Number of atoms:",N_at)

    batch_freq=int(nframes/args.batch_size) 

    ### ### ###

    # Calculate largest coordinate for generation.
    box_s=max_size(prmf,trajfs,'all',1.1)

    discriminator = DiscriminatorModel(N_at,args.N_classes) 
    generator = GeneratorModel(N_at,args.noise_dim,args.N_classes,box_s) 
    discriminator.to(device)
    generator.to(device)
    if (args.load_model==True and os.path.isfile(model_g_file) and os.path.isfile(model_d_file)):
        print("{} and {} loaded.".format(model_g_file,model_d_file))
        generator.load_state_dict(torch.load(model_g_file))
        discriminator.load_state_dict(torch.load(model_d_file))

    # For a conditional GAN, the loss function is the Binary Cross Entropy between the target and the input probabilities
    loss = nn.BCELoss() 

    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate) 
    generator_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    if args.train_mode:
        summary_writer=SummaryWriter(args.output_directory)
        
        # initialize lists for observables and graphs.
        e2e_distance=[]
        bonds_dev=[]
        angles_dev=[]
        losses_fig=plt.figure(1, figsize=(4, 4))
        e2e_fig=plt.figure(1, figsize=(4, 4))
        bonds_fig=[]
        angles_fig=[]
        for dl, d_label in enumerate(args.desired_labels):
            e2e_distance.append([])
            bonds_dev.append([])
            angles_dev.append([])
            bonds_fig.append(plt.figure(1, figsize=(4, 4)))
            angles_fig.append(plt.figure(1, figsize=(4, 4)))
        
        Loss_G_mean = []
        Loss_D_mean = []
        for epoch_idx in range(args.n_epochs): 
            G_loss = [] 
            D_loss = []    
            for batch_idx, data_input in enumerate(data_loader):
            
                # Generate noise and move it the device
                noise = torch.randn(args.batch_size,args.noise_dim).to(device) 
                # Forward pass
                fake_labels=torch.randint(0,args.N_classes,(args.batch_size,)).to(device)
                generated_data = generator(noise, fake_labels) 
                
                true_data = data_input[0].view(args.batch_size, 3*N_at).to(device) 
                digit_labels = data_input[1].to(device) 
                true_labels = torch.ones(args.batch_size).to(device) 

                # Clear optimizer gradients        
                discriminator_optimizer.zero_grad()
                # Forward pass with true data as input
                discriminator_output_for_true_data = discriminator(true_data,digit_labels).view(args.batch_size) 
                # Compute Loss
                true_discriminator_loss = loss(discriminator_output_for_true_data, true_labels) 
                
                # Forward pass with generated data as input
                discriminator_output_for_generated_data = discriminator(generated_data.detach(), fake_labels).view(args.batch_size) 
                # Compute Loss
                generator_discriminator_loss = loss(discriminator_output_for_generated_data, torch.zeros(args.batch_size).to(device)) 
                # Average the loss
                discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2 
                # Backpropagate the losses for Discriminator model.
                discriminator_loss.backward()
                discriminator_optimizer.step()
                D_loss.append(discriminator_loss.data.item())
                
                # Clear optimizer gradients
                generator_optimizer.zero_grad()        
                # It's a choice to generate the data again 
                generated_data = generator(noise, fake_labels) #.requires_grad_(False) 
                # Forward pass with the generated data
                discriminator_output_on_generated_data = discriminator(generated_data, fake_labels).view(args.batch_size) 
                # Compute loss: it must be the same of the discriminator, but reversed: the fake data must be passed as all ones, thus we use true_labels
                generator_loss = loss(discriminator_output_on_generated_data, true_labels) 
                # Backpropagate losses for Generator model.
                generator_loss.backward()
                generator_optimizer.step()
                G_loss.append(generator_loss.data.item())
                
                if (batch_idx==0 and epoch_idx==0): print("Initial: discriminator_loss: {} , generator_loss: {}".format(discriminator_loss.item(),generator_loss.item()))

                # Evaluate the model
                if ((batch_idx + 1)% batch_freq == 0 and (epoch_idx + 1)%args.epoch_freq == 0): 
                    with torch.no_grad(): 
                        noise = torch.randn(args.batch_size,args.noise_dim).to(device)
                        for dl, d_label in enumerate(args.desired_labels):
                            fake_labels = torch.tensor(args.batch_size*[dl]).to(device) 
                            generated_data = generator(noise, fake_labels).cpu().view(args.batch_size, 3*N_at) 
                            for x in generated_data:

                                # Generate .inpcrd file
                                outname='out_'+outmode+'_label'+str(fake_labels[0].item())+'_epoch'+str(last_epoch+epoch_idx+1)+'.inpcrd'
                                write_inpcrd(x.detach().numpy().reshape(N_at,3),outname=args.output_directory+outname)
                                if (epoch_idx+1)%args.log_freq==0: print("{} written.".format(outname))

                                # Calculate observables for later evaluation of the training
                                e2e_distance[dl].append([epoch_idx,check_label_condition(prmf,args.output_directory+outname)])
                                bonds_dev[dl].append([epoch_idx,bonds_deviation(prmf,args.output_directory+outname)])
                                angles_dev[dl].append([epoch_idx,angles_deviation(prmf,args.output_directory+outname)]) 
                                torch.save(generator.state_dict(),model_g_file) 
                                torch.save(discriminator.state_dict(),model_d_file) 
                                break
                    
            if (epoch_idx+1)%args.epoch_freq==0:
                summary_writer.add_scalar('Loss_d',torch.mean(torch.FloatTensor(D_loss)),global_step=epoch_idx)
                summary_writer.add_scalar('Loss_g',torch.mean(torch.FloatTensor(G_loss)),global_step=epoch_idx)
                Loss_D_mean.append([epoch_idx,torch.mean(torch.FloatTensor(D_loss))])
                Loss_G_mean.append([epoch_idx,torch.mean(torch.FloatTensor(G_loss))])

            if (epoch_idx+1)%args.log_freq==0:
                print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % ( (epoch_idx+last_epoch+1), args.n_epochs+last_epoch, torch.mean(torch.FloatTensor(D_loss)), torch.mean(torch.FloatTensor(G_loss))))

        # Plot loss averages over batches
        plt.plot(np.array(Loss_D_mean)[:, 0], np.array(Loss_D_mean)[:, 1],lw=1,c='C0',label='Loss D')
        plt.plot(np.array(Loss_G_mean)[:, 0], np.array(Loss_G_mean)[:, 1],lw=1,c='C1',label='Loss G')
        plt.legend(loc='upper right',prop={'size':15})
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        losses_fig.savefig(args.output_directory+'Losses.png',dpi=150)
        #plt.show()
        plt.clf()

        # Plot observables
        for dl, d_label in enumerate(args.desired_labels):
            plt.plot(np.array(bonds_dev[dl])[:, 0], np.array(bonds_dev[dl])[:, 1],lw=1,c='C0')
            plt.xlabel('Epoch')
            plt.ylabel('Bonds dev. [$\AA$]')
            bonds_fig[dl].savefig(args.output_directory+'Bonds_deviation_label'+str(d_label)+'.png',dpi=150)
            #plt.show()
            plt.clf()
            plt.plot(np.array(angles_dev[dl])[:, 0], np.array(angles_dev[dl])[:, 1],lw=1,c='C1')
            plt.xlabel('Epoch')
            plt.ylabel('Angle dev. [deg]')
            angles_fig[dl].savefig(args.output_directory+'Angles_deviation_label'+str(d_label)+'.png',dpi=150)
            #plt.show()
            plt.clf()

        # Plot the end-to-end distances
        for dl, d_label in enumerate(args.desired_labels):
            plt.plot(np.array(e2e_distance[dl])[:, 0], np.array(e2e_distance[dl])[:, 1],lw=1,c='C'+str(dl),label='Label '+str(d_label))
        plt.xlabel('Epoch')
        plt.ylabel('End-to-end distance [$\AA$]')
        plt.legend(loc='upper right',prop={'size':15})
        e2e_fig.savefig(args.output_directory+'End2end_distances.png',dpi=150)
        #plt.show()
        plt.clf()

    # Test mode
    else: 
        for structure_idx in range(args.n_structures):
            print("Generating structure:",structure_idx)
            with torch.no_grad(): 
                noise = torch.randn(args.batch_size,args.noise_dim).to(device)
                for dl,d_label in enumerate(args.desired_labels):
                    fake_labels = torch.tensor(args.batch_size*[dl]).to(device) 
                    generated_data = generator(noise, fake_labels).cpu().view(args.batch_size, 3*N_at) 
                    for x in generated_data:
                        outname='gen_'+outmode+'_label'+str(fake_labels[0].item())+'_'+str(structure_idx+1)+'.inpcrd' 
                        write_inpcrd(x.detach().numpy().reshape(N_at,3),outname=args.output_directory+outname)
                        print("{} written.".format(outname))
                        break
        

if __name__ == '__main__':
    main()




