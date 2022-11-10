#!/bin/bash

### ### BASH script for generation of input parameter+topology and trajectory ### ###

nconf=10000 #Number of desired conformations

outdir=../example_input # Output directory

rm $outdir/gen_mdcrd.cpptraj

# Loop over index of conformation
for idx in `seq 0 $((nconf-1))`
do
rm leap.log
echo $idx | awk '($1%100==0){print $1}'
# Generate bonds, angles, and dihedrals randomly in valid range
val0=$( echo "scale=10; $RANDOM/32767*180" | bc )
val1=$( echo "scale=10; $RANDOM/32767*180" | bc )

# Make LEaP input script
echo \
'source leaprc.gaff2
source leaprc.protein.ff14SB 
source leaprc.water.tip3p 
loadamberparams frcmod.ionsjc_tip3p

UN = sequence { THR ALA GLY GLY LYS SER }

impose UN { {1 6} } { { "N" "CA" "C" "N" '$val0' } { "C" "N" "CA" "C" '$val1' } }

check UN

saveamberparm UN '$outdir'/peptide.prmtop '$outdir'/peptide_'$idx'.inpcrd

quit
' > leap.in

# Execute LEaP
tleap -s -f leap.in &> /dev/null
grep -e 'Errors = ' leap.log | grep -v 0 && echo "ERROR in LEaP." | exit

# Append line to cpptraj script
echo 'trajin '$outdir'/peptide_'$idx'.inpcrd' >> $outdir/gen_mdcrd.cpptraj
done 

# Collate all structures into a single trajectory file
echo "trajout "$outdir/"all_conformations.mdcrd mdcrd" >> $outdir/gen_mdcrd.traj
echo "center @CA,C,N origin" >> $outdir/gen_mdcrd.traj
echo "distance dist1 :1@CA :6@CA out $outdir/dist.dat" >> $outdir/gen_mdcrd.traj
echo "hist dist1 bins 50 out $outdir/hist.dat" >> $outdir/gen_mdcrd.traj
echo "go" >> $outdir/gen_mdcrd.traj
cpptraj -p $outdir/peptide.prmtop < $outdir/gen_mdcrd.traj > /dev/null

rm $outdir/peptide*.inpcrd

echo "Successfully put $idx structures in $outdir/all_conformations.mdcrd"

