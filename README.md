FrameGen is a conditional Generative Adversarial Network (cGAN) which, given the AMBER parameter and topology file of a biological system (.prmtop, .top) and an associated molecular dynamics (MD) trajectory (.dcd, .nc, ...), reconstructs a frame belonging to the same distribution of the sampled trajectory. Moreover, it is given a label (0 or 1) which allow to distinguish between frames with an observed condition from those without.

This code is based on the cGAN example provided at this link: https://medium.com/analytics-vidhya/step-by-step-implementation-of-conditional-generative-adversarial-networks-54e4b47497d6

The trajectory is aligned and centered around the origin in the pre-processing stage, as this is necessary for the algorithm. The first frame is outputted as "initial.inpcrd"

Both the generator and the discriminator make use of a Binary Cross Entropy function (as standard for GANs) as a loss function.

The generator takes as input gaussian random noise of custom size (default: 100). Its output is 3N dimensional, with N being the number of atoms, and is written in an .inpcrd file according to AMBER format.

The activation function for the output layer of the generator is chosen as hyperbolic tangent, multiplied by the maximum size max_s of the system. This consist in the largest absolute x, y, or z coordinate of any atom during trajectory (multiplied by 1.1 in order to allow for fluctuations). Thus, the output is guaranteed to be generated within -max_s and +max_s in every direction.

The library ParmEd (which is part of the package AmberTools) is used to check the maximum deviation of bonds and angles from their equilibrium values, according to the force-field used for parametrisation (ff14SB)

# Required libraries
PyTORCH modules required: ./libraries_versions.sh
MDAnalysis : 2.2.0
torch : 1.12.1+cu116
numpy : 1.22.3
ParmEd: 3.4.3

# Demonstrative Run

As a case-study, the software LEaP and CPPTRAJ (AmberTools) were used to generate 10,000 conformations of a simply peptide (sequence TAGGKS), and arranged in a trajectory file. Approximately half of the frames satisfy the condition of end-to-end distance smaller than 1 nm, whereas the other half do not. This was chosen as the condition to feed to the cGAN.

A Ramachandran plot of the phi and psi angles is also checked against that of the initial frame.

# Visualization
The output .inpcrd files can be visualized with VMD (or pymol): load the parameter and topology files as "AMBER7 Parm", then the .incprd as "AMBER7 Restart"

# Further improvements

1) The architectures of both the Generator and the Discriminator likely depend on the number of atoms of the system.
2) The input noise dimension is also likely to depend on the number of atoms.
3) The discriminator may be fed with collective variables, such as bonds and angles, in order to improve the quality of the generated structures. 


