#!/bin/bash
#SBATCH --job-name=run_amd_gpu_other_rocm
#SBATCH --output=run_amd_gpu_other_rocm.%j.out
#SBATCH --error=run_amd_gpu_other_rocm.%j.err
#SBATCH --partition=ami100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH --account=amc-general
#SBATCH --reservation=amc_workshop
#SBATCH --time=00:10:00


# We make sure that the user can run mambaforge
cp ~/.condarc ~/.mambarc

echo "*********************************** Loading ROCM and pytorch + exporting useful paths  ********************************"
# We load ROCM which is the collection of drivers, libraries, tools and API to run 
# on an AMD GPU : https://rocm.docs.amd.com/en/latest/what-is-rocm.html
# For NVIDIA it is NVIDIA CUDA

echo "*********************************** Loading the latest ROCM on Alpine and Python ********************************"
module load rocm/5.5.0
# We load python from Alpine
module load python

# We export the TMP_DIR so that we do not fill /tmp space
export TMP=/gpfs/alpine1/scratch/$USER/cache_dir
mkdir -pv $TMP
export TEMP=$TMP
export TMPDIR=$TMP
export TEMPDIR=$TMP
export PIP_CACHE_DIR=$TMP


# We point the pip cache dir to /scratch as well
export PIP_CACHE_DIR=/gpfs/alpine1/scratch/$USER/cache_dir
mkdir $PIP_CACHE_DIR

# We determine an install directory, in scratch for this 
export PIP_INSTALL_DIR=/gpfs/alpine1/scratch/$USER/software/rocm_5_5_0_install
mkdir -pv $PIP_INSTALL_DIR

# We export all the ROCM related path and libraries
#export ROCM_PATH=/curc/sw/install/rocm/5.2.3
#export HIP_PATH=$ROCM_PATH
#export HIP_ROOT_DIR=$ROCM_PATH
#export ROCM_LIBRARIES=$ROCM_PATH/lib
export PYTHONPATH=$PIP_INSTALL_DIR:$PIP_INSTALL_DIR/bin:$PYTHONPATH
export PATH=$PIP_INTALL_DIR:$PATH
#export DEVICE_LIB_PATH=/curc/sw/install/rocm/5.2.3/amdgcn/bitcode
#export ROCM_DEVICE_LIB_PATH=/curc/sw/install/rocm/5.2.3/amdgcn/bitcode

# We need to specify the HIP platform
HIP_PLATFORM=amd
USE_ROCM=1
# We may now install the additional pytorch AMD compatible packages:
export PATH=$PATH:$PIP_INSTALL_DIR/bin

# Running the vision transformer example
# This example shows how we fit a 3rd order polynomial to a sine function: Example from  https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

echo "*********************************** Running the models  ********************************"
# We want to make sure that we can profile what is going on
# We want to set AMD_LOG_LEVEL for profiling and debugging
#CODE 0 means No log
#CODE 1 means log the error
#CODE 2 means log warnings 
#CODE 3 means log information
#CODE 4 means debug mode
# More information here: https://rocm.docs.amd.com/projects/HIP/en/develop/developer_guide/logging.html

HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 AMD_LOG_LEVEL=1 HSA_ENABLE_SDMA=0 python tensors.py

