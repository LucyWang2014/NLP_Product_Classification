#!/bin/bash

#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=10:00:00
#PBS -l mem=23GB
#PBS -N test_img_process_67_200k
#PBS -j oe

THEANO_FLAGS='floatX=float32,device=gpu,cuda.root=/share/apps/cuda/6.5.12'
if [ "$PBS_JOBTMP" != "" ]; then
        THEANO_FLAGS="base_compiledir=$PBS_JOBTMP,$THEANO_FLAGS"
fi

export THEANO_FLAGS

cd /scratch/cdg356/spring/scripts

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

python data_prep.py
