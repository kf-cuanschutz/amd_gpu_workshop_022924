#!/bin/bash

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
export PIP_INSTALL_DIR=/projects/$USER/software/pip_torch_amd_install
mkdir -pv $PIP_INSTALL_DIR

# We export all the ROCM related path and libraries
export ROCM_PATH=/curc/sw/install/rocm/5.2.3
export HIP_PATH=$ROCM_PATH
export HIP_ROOT_DIR=$ROCM_PATH
export ROCM_LIBRARIES=$ROCM_PATH/lib
export PYTHONPATH=$PIP_INSTALL_DIR:$PIP_INSTALL_DIR/bin:$PYTHONPATH
export PATH=$PIP_INTALL_DIR:$PATH
export DEVICE_LIB_PATH=/curc/sw/install/rocm/5.2.3/amdgcn/bitcode
export ROCM_DEVICE_LIB_PATH=/curc/sw/install/rocm/5.2.3/amdgcn/bitcode

# We need to specify the HIP platform
HIP_PLATFORM=amd
USE_ROCM=1

# We may now install the additional pytorch AMD compatible packages:
export PATH=$PATH:$PIP_INSTALL_DIR/bin

# We may now start installing horovod
echo "********** Environment deactivated *************"
module unload mambaforge

export GXX_PATH=/projects/$USER/software/anaconda/envs/gxx_horovod_amd
export LD_LIBRARY_PATH=$LIBRARY_PATH:$GXX_PATH/lib
export PATH=$PATH:$GXX_PATH/bin:$GXX_PATH/include

# 2) Loading the appropriate module dependencies for Horovod:
module load gcc openmpi cuda cmake

# 3) Exporting the appropriate NCCL related libraries:
#export HOROVOD_NCCL_INCLUDE=/projects/kefo9343/software/spack/opt/spack/linux-rhel8-zen/gcc-8.4.1/nccl-2.18.3-1-2pmuetqeeecftln5wn6jqy6xcvyzak6d/include
#export HOROVOD_NCCL_LIB=/projects/kefo9343/software/spack/opt/spack/linux-rhel8-zen/gcc-8.4.1/nccl-2.18.3-1-2pmuetqeeecftln5wn6jqy6xcvyzak6d/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/kefo9343/software/spack/opt/spack/linux-rhel8-zen/gcc-8.4.1/nccl-2.18.3-1-2pmuetqeeecftln5wn6jqy6xcvyzak6d/lib

export HOROVOD_RCCL_INCLUDE=/curc/sw/install/rocm/5.2.3/include:/curc/sw/install/rocm/5.2.3/rccl/include
export HOROVOD_RCCL_LIB=/curc/sw/install/rocm/5.2.3/lib:/curc/sw/install/rocm/5.2.3/rccl/lib
export HOROVOD_RCCL_HOME=/curc/sw/install/rocm/5.2.3/rccl

# 4) Installing Horovod
export HOROVOD_INSTALL_DIR=$PIP_INSTALL_DIR          #/projects/$USER/software/horovod_amd_install
export HOROVOD_CMAKE_INSTALL_PREFIX=$HOROVOD_INSTALL_DIR


# Exporting the install directory
export PATH=$PATH:$HOROVOD_CMAKE_INSTALL_PREFIX/bin

# Submit horovod:

# To run it on multiple nodes, I will need to know their hostnames
scontrol show hostname > $SLURM_SUBMIT_DIR/nodelist.txt
export SLURM_NODEFILE=$SLURM_SUBMIT_DIR/nodelist.txt
node1=`sed -n 1p $SLURM_SUBMIT_DIR/nodelist.txt`

# If you want to add a new node you run the following line number:
node2=`sed -n 2p $SLURM_SUBMIT_DIR/nodelist.txt`


# We will time the pipeline
echo "*************Running Horovod ******************"

start=`date +%s.%N`

#CUDA_VISIBLE_DEVICES=4 HOROVOD_WITH_PYTORCH=1 \
#HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_GPU=ROCM \
#HOROVOD_ROCM_HOME=$ROCM_PATH HOROVOD_WITH_MPI=1 \
#AMD_LOG_LEVEL=1 ROC_ENABLE_PRE_VEGA=1  AMD_LOG_LEVEL=1 \
#AMDGPU_TARGETS="gfx803" HSA_OVERRIDE_GFX_VERSION=10.3.0  mpirun -machinefile $SLURM_NODEFILE -np 4  python pytorch_mnist.py   #horovodrun --verbose -np 4 -H $node1:2,$node2:2  pytorch_mnist.py


#__HIP_PLATFORM_AMD__=1  HOROVOD_CPU_OPERATIONS=MPI PYTORCH_ROCM_ARCH=gfx908 GFX_ARCH=gfx908 \
#HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 HOROVOD_WITH_PYTORCH=1 
#PYTORCH_ROCM_ARCH=gfx908 \
#	HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_GPU=ROCM \
#	HOROVOD_ROCM_HOME=$ROCM_PATH HOROVOD_WITH_MPI=1 \
#	AMD_LOG_LEVEL=4 ROC_ENABLE_PRE_VEGA=1  AMD_LOG_LEVEL=1 \
#	AMDGPU_TARGETS="gfx908" HSA_OVERRIDE_GFX_VERSION=10.3.0  horovodrun -np 1 -H localhost:1  python pytorch_mnist.py --communication nccl


HSA_ENABLE_SDMA=0 OFFLOAD_ARCH="gfx908" \
AMD_LOG_LEVEL=5 ROC_ENABLE_PRE_VEGA=1 HSA_OVERRIDE_GFX_VERSION=9.8.0 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
CMAKE_HIP_ARCHITECTURES="gfx908"  CMAKE_HIP_PLATFORM="gfx908" ROCM_VERSION=5.2.3 \
AMDGPU_TARGET="gfx908" \
__HIP_PLATFORM_AMD__=1  HOROVOD_CPU_OPERATIONS=MPI PYTORCH_ROCM_ARCH="gfx908" \
AMDGPU_TARGETS="gfx908" GFX_ARCH="gfx908" HAVE_GPU=1  HOROVOD_WITH_PYTORCH=1 HAVE_ROCM=1 Pytorch_ROCM=1 \
HOROVOD_GPU_ALLREDUCE=NCCL  HOROVOD_GPU=ROCM HOROVOD_ROCM_HOME=$ROCM_PATH HOROVOD_WITH_MPI=1 \
horovodrun -np 1 -H localhost:1  python pytorch_mnist.py 

end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "############# The runtime is $runtime ##################"
