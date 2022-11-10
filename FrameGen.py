import os
import sys
sys.path.append('./src/')
from functions import *
from models import *
from plots import *
#
from tensorboardX import SummaryWriter  
#
import argparse

parser = argparse.ArgumentParser()

# Mode settings
parser.add_argument('--train_mode', default=True, type=bool) # Set to False for Test mode.
parser.add_argument('--load_model', default=False, type=bool)

# Classes settings
parser.add_argument('--dist_cut', default=10.0, type=float) # distance cut-off for class defintion
parser.add_argument('--N_classes', default=2, type=int) # Number of classes
parser.add_argument('--desired_labels', default=[0,1], type=list) # Classes to be considered for output

# Bio-system settings
parser.add_argument('--biosystem', default='PROTEIN', type=str) # Necessary for atom selections (see below)

# Model settings
parser.add_argument('--n_epochs', default=1000, type=int) # Number of epochs for training
parser.add_argument('--batch_size', default=100, type=int) # Batch size for training
parser.add_argument('--learning_rate', default=0.0002, type=float) # Learning rate for Adam optimizer
parser.add_argument('--noise_dim', default=100, type=int) # Dimension for gaussian noise to feed to the generator

# Output settings
parser.add_argument('--desired_format', default='inpcrd', type=str)
parser.add_argument('--epoch_freq', default=10, type=int) # Output will be generated every this many epochs
parser.add_argument('--log_freq', default=100, type=int) # Verbose output will be printed every this many epochs
parser.add_argument('--n_structures', default=10, type=int) # Number of structures to be generated for every class (only in Test mode)
parser.add_argument('--input_directory', default='./example_input/' , type=str) 
parser.add_argument('--output_directory', default='./example_output/', type=str) 

args = parser.parse_args()

prmf = args.input_directory+'peptide.prmtop' # Parameter and topology file
#trajfs = [args.input_directory+'all_conformations.mdcrd'] # Trajectory files list
trajfs = ['/home/acrnjar/Desktop/TEMP/Peptides_gen/all_conformations.mdcrd'] # Trajectory files list

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cuda is available:",torch.cuda.is_available())

def main():

    print("Output will be written in:",args.output_directory)
    if not os.path.exists(args.output_directory): os.system('mkdir '+args.output_directory)

    if args.biosystem=='PROTEIN':
        backbone='name CA C N'
    elif args.biosystem=='DNA':
        backbone='name P'

    print("train_mode:",args.train_mode)
    print("load_model:",args.load_model)

    model_g_file=args.output_directory+'model_generator.pth'
    model_d_file=args.output_directory+'model_discriminator.pth'

    # Remove previous output if necessary
    if args.train_mode:
        os.system('rm '+args.output_directory+'gen*'+args.desired_format+' 2> /dev/null')
        if not args.load_model:
            os.system('rm '+args.output_directory+'out*'+args.desired_format+' 2> /dev/null')

    # Determine last saved epoch   
    last_epoch=0
    if args.train_mode and args.load_model:
        prefixed = [filename for filename in os.listdir(args.output_directory) if filename.startswith("out_train_label1_epoch")]
        past_epochs=[]
        for wfile in prefixed:
            past_epochs.append(int(wfile.replace('out_train_label1_epoch','').replace('.'+desired_format,'')))
        last_epoch=max(past_epochs)
        print("Last epoch found:",last_epoch)

    # Define MDAnalysis universe and related parameters
    univ = mda.Universe(prmf, trajfs)
    nframes = len(univ.trajectory)
    batch_freq = int(nframes/args.batch_size) 
    N_at = len(univ.select_atoms('all'))
    box_s = max_size(prmf,trajfs,'all',1.1) # Calculate largest coordinate for generation

    # Generate data
    dataset, atoms_list = generate_training_data(prmf, trajfs, 0, nframes-(nframes%args.batch_size), backbone, args.dist_cut, args.output_directory)

    # Define discriminator and generator models
    discriminator = DiscriminatorModel(N_at, args.N_classes, n1=50, n2=100, n3=200) 
    generator = GeneratorModel(N_at, args.noise_dim, args.N_classes, box_s, n1=200, n2=100, n3=50) 
    discriminator.to(device)
    generator.to(device)
    if (args.load_model==True and os.path.isfile(model_g_file) and os.path.isfile(model_d_file)):
        print("{} and {} loaded.".format(model_g_file,model_d_file))
        generator.load_state_dict(torch.load(model_g_file))
        discriminator.load_state_dict(torch.load(model_d_file))

    # For a conditional GAN, the loss function is the Binary Cross Entropy between the target and the input probabilities
    loss = nn.BCELoss() 

    # Define optimizers
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate) 
    generator_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate)

    # Load data
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.train_mode:
        summary_writer = SummaryWriter(args.output_directory)
        
        # initialize lists for observables and graphs.
        e2e_distance = []
        bonds_dev = []
        angles_dev = []
        
        for dl, d_label in enumerate(args.desired_labels):
            e2e_distance.append([])
            bonds_dev.append([])
            angles_dev.append([])
        
        Loss_G_mean = []
        Loss_D_mean = []
        for epoch_idx in range(args.n_epochs): 

            G_loss = [] 
            D_loss = []    
            for batch_idx, data_input in enumerate(data_loader):
            
                loss_d, loss_g = training(discriminator, discriminator_optimizer, generator, generator_optimizer, loss, data_input, args.batch_size, args.noise_dim, args.N_classes, N_at)
                G_loss.append(loss_g)
                D_loss.append(loss_d)

                if (batch_idx==0 and epoch_idx==0): print("Initial: discriminator_loss: {} , generator_loss: {}".format(loss_d, loss_g))

                # Evaluate the model
                if ((batch_idx + 1)% batch_freq == 0 and (epoch_idx + 1)%args.epoch_freq == 0): 
                    with torch.no_grad(): 
                        noise = torch.randn(args.batch_size,args.noise_dim).to(device)
                        for dl, d_label in enumerate(args.desired_labels):
                            fake_labels = torch.tensor(args.batch_size*[dl]).to(device) 
                            generated_data = generator(noise, fake_labels).cpu().view(args.batch_size, 3*N_at) 
                            for x in generated_data:

                                # Generate .inpcrd file
                                outname='out_label'+str(fake_labels[0].item())+'_epoch'+str(last_epoch+epoch_idx+1)+'.inpcrd'
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
        plot_losses(Loss_D_mean, Loss_G_mean, args.output_directory)

        # Plot observables
        for dl, d_label in enumerate(args.desired_labels):
            plot_observables(d_label, bonds_dev[dl], angles_dev[dl], args.output_directory)

        # Plot the end-to-end distances
        plot_e2e(e2e_distance, args.desired_labels, args.output_directory)

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
                        outname='gen_label'+str(fake_labels[0].item())+'_'+str(structure_idx+1)+'.inpcrd' 
                        write_inpcrd(x.detach().numpy().reshape(N_at,3),outname=args.output_directory+outname)
                        print("{} written.".format(outname))
                        break
        

if __name__ == '__main__':
    main()




