import numpy as np
import matplotlib.pyplot as plt

def plot_losses(Loss_D_mean, Loss_G_mean, output_directory):

    losses_fig = plt.figure(1, figsize=(4, 4))
    plt.plot(np.array(Loss_D_mean)[:, 0], np.array(Loss_D_mean)[:, 1],lw=1,c='C0',label='Discriminator')
    plt.plot(np.array(Loss_G_mean)[:, 0], np.array(Loss_G_mean)[:, 1],lw=1,c='C1',label='Generator')
    plt.legend(loc='upper right',prop={'size':15})
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    losses_fig.savefig(output_directory+'Losses.png',dpi=150)
    #plt.show()
    plt.clf()

def plot_observables(d_label, bonds_dev, angles_dev, output_directory):

    bonds_fig = plt.figure(1, figsize=(4, 4))
    plt.plot(np.array(bonds_dev)[:, 0], np.array(bonds_dev)[:, 1],lw=1,c='C0')
    plt.xlabel('Epoch')
    plt.ylabel('Bonds dev. [$\AA$]')
    bonds_fig.savefig(output_directory+'Bonds_deviation_label'+str(d_label)+'.png',dpi=150)
    #plt.show()
    plt.clf()

    angles_fig = plt.figure(1, figsize=(4, 4))
    plt.plot(np.array(angles_dev)[:, 0], np.array(angles_dev)[:, 1],lw=1,c='C1')
    plt.xlabel('Epoch')
    plt.ylabel('Angle dev. [deg]')
    angles_fig.savefig(output_directory+'Angles_deviation_label'+str(d_label)+'.png',dpi=150)
    #plt.show()
    plt.clf()

def plot_e2e(e2e_distance, desired_labels, output_directory):

    e2e_fig = plt.figure(1, figsize=(4, 4))
    for dl, d_label in enumerate(desired_labels):
        plt.plot(np.array(e2e_distance[dl])[:, 0], np.array(e2e_distance[dl])[:, 1],lw=1,c='C'+str(dl),label='Label '+str(d_label))
    plt.xlabel('Epoch')
    plt.ylabel('End-to-end distance [$\AA$]')
    plt.legend(loc='upper right',prop={'size':15})
    e2e_fig.savefig(output_directory+'End2end_distances.png',dpi=150)
    #plt.show()
    plt.clf()