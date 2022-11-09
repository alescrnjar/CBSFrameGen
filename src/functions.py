import torch
import torch.nn as nn
from torch import optim as optim
#
import MDAnalysis as mda
from MDAnalysis.analysis import align
from numpy.linalg import norm
#
import numpy as np
#
import parmed

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
    
def generate_training_data(prm_top_file, traj_file, frame_i, frame_f, backbone, dist_cut, output_dir):
    """
    Generate training dataset.
    """
    u = mda.Universe(prm_top_file, traj_file)
    input_dats = []
    count_0 = 0
    count_1 = 0
    at_list = []
    listed = False

    ref_u = u
    ref_u.trajectory[0]
    ref_pos = ref_u.select_atoms(backbone).positions - ref_u.atoms.center_of_mass() #backbone works with both proteins and DNA
    
    for ts in u.trajectory[frame_i:frame_f:1]:

        # Align the current frame to the first one
        prot_pos = u.select_atoms(backbone).positions - u.atoms.center_of_mass()
        R_matrix, R_rmsd = align.rotation_matrix(prot_pos,ref_pos)
        u.atoms.translate(-u.select_atoms(backbone).center_of_mass())
        u.atoms.rotate(R_matrix)
        u.atoms.translate(ref_u.select_atoms(backbone).center_of_mass())

        sel = u.select_atoms('all')

        # Make atom list with atom names and coordinates
        if not listed:
            for atx in range(len(sel.atoms.ids)):
                at_sel=u.select_atoms('bynum '+str(atx+1))
                at_list.append([])
                at_list[-1].append(at_sel.residues.resids[0])
                at_list[-1].append(at_sel.residues.resnames[0])
                at_list[-1].append(at_sel.atoms.names[0])
                at_list[-1].append(at_sel.atoms.ids[0])
                listed = True

        # Define observable for labeling data
        pos_dstz_at1 = u.select_atoms('resid 1 and name CA').center_of_mass()
        pos_dstz_at2 = u.select_atoms('resid 6 and name CA').center_of_mass()
        dist_dstz = norm(pos_dstz_at1-pos_dstz_at2)

        # Assign label 
        if (dist_dstz < dist_cut): 
            lab_val = 1
            count_1 += 1
        else:
            lab_val = 0
            count_0 += 1
        
        input_dats.append((torch.tensor(sel.positions),lab_val)) 
        if (ts.frame == 0):
            write_inpcrd(sel.positions,outname=output_dir+'initial.inpcrd')
    input_dataset=input_dats
    print("{} frames with label 0, {} frames with label 1.".format(count_0,count_1))
    return input_dataset,at_list
