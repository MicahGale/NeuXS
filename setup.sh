#!/usr/bin/env bash

# Euler doesn't have hdf5 installed in it. So we need to download and compile it
# And for a weird reason HDF5 lib from the website doesn't compile
# My work around that is just installing it from the source
# Note to Dan or TAs:
#  Just clone this repo and run this bash script
#  Even if doesn't work please let us know. We have tested this in Euler and
#  "This works on my machine. IDK why it isn't working on you machine"

HDF5_lib_dir="${HOME}/hdf5lib"
NeuXS=${PWD}
mkdir -p $HDF5_lib_dir


# loading modules
module load nvidia/cuda/13.0.0
module load gcc/13.2.0
module load cmake/4.1.2

# Installing hdf5 in a custom dir as I don't want get yelled by Euler folks
cd ${HOME}
git clone https://github.com/HDFGroup/hdf5.git
cd hdf5 && mkdir -p build && cd  build
cmake -DCMAKE_INSTALL_PREFIX=$HDF5_lib_dir ..
make -j$(nproc)
make install

# hdf issues are done. Now install and unzip the cross sections
# this one downloads ENDF-VIII.0 data. If you change the version be sure to change
# OPENMC_CROSS_SECTIONS ENV as well.

cd ${HOME}
wget https://anl.box.com/shared/static/uhbxlrx7hvxqw27psymfbhi7bx7s6u6a.xz
tar xf uhbxlrx7hvxqw27psymfbhi7bx7s6u6a.xz
rm uhbxlrx7hvxqw27psymfbhi7bx7s6u6a.xz


# set the cross section variable
echo "export OPENMC_CROSS_SECTIONS='${HOME}/endfb-viii.0-hdf5/cross_sections.xml'" >> $HOME/.bashrc
source ~/.bashrc

# we aren't doing too much fancy stuff from c++ side so gcc 13.2.0 vs gcc 13.3.0 are almost same
cd NeuXS && mkdir -p build && cd build
cmake .. -DHDF5_ROOT=$HDF5_lib_dir -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler"
make -j $nproc
# Now that the program is compiled you can use the job_script in scripts dir to
# test the bench mark.

