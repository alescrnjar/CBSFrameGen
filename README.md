# FrameGen

FrameGen is a conditional Generative Adversarial Network (cGAN) which, given the AMBER parameter and topology file of a biological system (.prmtop, .top) and an associated molecular dynamics (MD) trajectory (.dcd, .nc, ...), reconstructs a frame belonging to the same distribution of the sampled trajectory. Moreover, it is given a label (0 or 1) which allow to distinguish between frames with an observed condition from those without.

This code is based on the cGAN example provided at this link: https://medium.com/analytics-vidhya/step-by-step-implementation-of-conditional-generative-adversarial-networks-54e4b47497d6

The trajectory is aligned and centered around the origin in the pre-processing stage, as this is necessary for the algorithm. The first frame is outputted as "initial.inpcrd"

Both the generator and the discriminator make use of a Binary Cross Entropy function (as standard for GANs) as a loss function.

The generator takes as input gaussian random noise of custom size (default: 100). Its output is 3N dimensional, with N being the number of atoms, and is written in an .inpcrd file according to AMBER format.

The activation function for the output layer of the generator is chosen as hyperbolic tangent, multiplied by the maximum size box_s of the system. This consist in the largest absolute x, y, or z coordinate of any atom during trajectory (multiplied by 1.1 in order to allow for fluctuations). Thus, the output is guaranteed to be generated within -box_s and +box_s in every direction.

The library ParmEd (which is part of the package AmberTools) is used to check the maximum deviation of bonds and angles from their equilibrium values, according to the force-field used for parametrisation (ff14SB).

The output .inpcrd files can be visualized with VMD (or pymol): load the parameter and topology files as "AMBER7 Parm", then the .incprd as "AMBER7 Restart"

# Required Libraries

Python modules required: 

* numpy >= 1.22.3

* torch >= 1.12.1+cu116

* MDAnalysis >= 2.2.0

* ParmEd >= 3.4.3 

* tensorboardX >= 2.5.1

# Modules

* models.py : contains PyTorch classes of the discriminator and the generator.
* functions.py : contains functions to manipulate data with MDAnalysis.
* plots.py : contains functions to produce output graphs.

# Case Study

As a case-study, the software LEaP and CPPTRAJ (AmberTools21) were used to generate 10,000 conformations of a simply peptide (sequence TAGGKS), and arranged in a trajectory file. Approximately half of the frames satisfy the condition of end-to-end distance smaller than 1 nm, whereas the other half do not. This was chosen as the condition to feed to the cGAN.

![alt text](https://github.com/alescrnjar/FrameGen/blob/main/example_output/Initial_Label0_Label1.png)

![alt text](https://github.com/alescrnjar/FrameGen/blob/main/example_output/Losses.png)



