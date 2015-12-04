#!/bin/bash

#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=50:00:00
#PBS -l mem=6GB
#PBS -N img_process
#PBS -j oe

cd /home/cdg356/spring/scripts

module purge
module load pillow/intel/2.7.0
module load pandas/intel/0.16.0
module load cuda/6.5.12
module load cudnn/6.5
module load theano/20150721
module load lasagne/20151007
module load scikit-image/intel/20150129

export LD_LIBRARY_PATH=/share/apps/cudnn/6.5/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/share/apps/cudnn/6.5/lib64:$LIBRARY_PATH
export CPATH=/share/apps/cudnn/6.5/include:$CPATH

env THEANO_FLAGS='floatX=float32,device=gpu,cuda.root=/share/apps/cuda/6.5.12' python data_prep.py