#!/bin/bash

# This script contains the steps necessary to 
# run the AMD container on Alpine
# Derived from https://rocm.docs.amd.com/projects/install-on-linux/en/develop/how-to/3rd-party/pytorch-install.html

#I am putting in comment the command I used to download and run the container:
# Make sure to comment all the export TMP in your ~/.bashrc
# apptainer build dev-ubuntu-20.04_latest.sif docker://rocm/dev-ubuntu-20.04
# apptainer overlay create --fakeroot --sparse --size 100000 sparse_3_simple_overlay.img
# Make sure to comment all the export TMP in your ~/.bashrc

export ALPINE_SCRATCH=/gpfs/alpine1/scratch/$USER
export APPTAINER_TMPDIR=$ALPINE_SCRATCH/singularity/tmp
export APPTAINER_CACHEDIR=$ALPINE_SCRATCH/singularity/cache
mkdir -pv $APPTAINER_CACHEDIR $APPTAINER_TMPD


echo "********************************* Copying the .sif and .img files  *************************"
cp /scratch/alpine/kfotso@xsede.org/test_container_docker/sparse_3_simple_overlay.img .
cp /scratch/alpine/kfotso@xsede.org/test_container_docker/build-dev-ubuntu-20.04_latest.sif .
export CONTAIN_DIR=${PWD}

echo "********************************* Running the built container *************************"
# --containall is very important as it allows to transfer video permission to the container
# Otherwise you will not see the AMD GPU
#--overlay allows to create a filesystem on top of the immutable .sif container
# so that you can modify it at will with the --fakeroot option
# To install packages and run pipeline inside the container we run:
apptainer shell -H $CONTAIN_DIR --bind=/dev/kfd,/dev/dri --fakeroot --rocm --containall --overlay sparse_3_simple_overlay.img build-dev-ubuntu-20.04_latest.sif


