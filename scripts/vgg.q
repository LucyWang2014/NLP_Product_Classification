#!/bin/bash

#PBS -q cuda
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=25:00:00
#PBS -l mem=8GB
#PBS -N vgg1
#PBS -j oe

cd /home/cdg356/spring/scripts

module purge
module load cuda/6.5.12
module load cudnn/6.5
module load theano/20150721
module load lasagne/20151007
module load scikit-image/intel/20150129
module load pillow/intel/2.7.0

export LD_LIBRARY_PATH=/share/apps/cudnn/6.5/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/share/apps/cudnn/6.5/lib64:$LIBRARY_PATH
export CPATH=/share/apps/cudnn/6.5/include:$CPATH

env THEANO_FLAGS='floatX=float32,device=gpu,cuda.root=/share/apps/cuda/6.5.12' python vgg_pretrained.py
