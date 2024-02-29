#!/bin/bash
#SBATCH --job-name=run_amd_gpu_rocprof
#SBATCH --output=run_amd_gpu_rocprof.%j.out
#SBATCH --error=run_amd_gpu_rocprof.%j.err
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
module load rocm/5.2.3

# We load pytorch 1.13.0 that has been built for the ROCM,
# which is compatible with ROCM5.2.X : https://pytorch.org/get-started/previous-versions/
module load pytorch/1.13.0

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
export PIP_INSTALL_DIR=/gpfs/alpine1/scratch/$USER/software/amd_gpu_standard_install
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

# Now we try roc_prof which is the equivalent of nvprof. More info here: https://extremecomputingtraining.anl.gov/wp-content/uploads/sites/96/2022/11/ATPESC-2022-Track-6-Talk-4-Tandon-AMD.pdf
rocprof  --stats --basenames --hip-trace python tensors.py


