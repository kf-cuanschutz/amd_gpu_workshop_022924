#!/bin/bash

# We make sure that the user can run mambaforge
cp ~/.condarc ~/.mambarc


# We load ROCM which is the collection of drivers, libraries, tools and API to run 
# on an AMD GPU : https://rocm.docs.amd.com/en/latest/what-is-rocm.html
# For NVIDIA it is NVIDIA CUDA

echo "*********************************** Loading the latest ROCM on Alpine and Python ********************************"
module load rocm/5.5.0

# We load python from Alpine
module load python

# We export the TMP_DIR so that we do not fill /tmp space
echo "*********************************** Exporting TMP related directories  ********************************"
export TMP=/gpfs/alpine1/scratch/$USER/cache_dir
mkdir -pv $TMP
export TEMP=$TMP
export TMPDIR=$TMP
export TEMPDIR=$TMP
export PIP_CACHE_DIR=$TMP


# We point the pip cache dir to /scratch as well
echo "*********************************** Exporting PIP related variables  ********************************"
export PIP_CACHE_DIR=/gpfs/alpine1/scratch/$USER/cache_dir
mkdir $PIP_CACHE_DIR

# We determine an install directory, in scratch for this 
export PIP_INSTALL_DIR=/gpfs/alpine1/scratch/$USER/software/rocm_5_5_0_install
mkdir -pv $PIP_INSTALL_DIR

# We determine an install directory, in scratch for this
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

echo "*********************************** Installing complementary packages  ********************************"
# We may now install the additional pytorch AMD compatible packages:
# We install pytorch 2.2.0 that has been built for the ROCM,
# which is compatible with ROCM5.6 : https://pytorch.org/get-started/previous-versions/
pip install --target=$PIP_INSTALL_DIR torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/rocm5.6
pip install --target=$PIP_INSTALL_DIR pandas
pip install --target=$PIP_INSTALL_DIR transformers
pip install --target=$PIP_INSTALL_DIR scikit-learn
#pip install --target=$PIP_INSTALL_DIR cmake
pip install --target=$PIP_INSTALL_DIR tqdm
pip install --target=$PIP_INSTALL_DIR  mxnet #mxnet is a deeplearning framework for efficiency and flexibility : https://pypi.org/project/mxnet/
export PATH=$PATH:$PIP_INSTALL_DIR/bin


