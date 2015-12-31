#!/bin/bash

#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=20:00:00
#PBS -l mem=40GB
#PBS -N omain_200k_multi
#PBS -j oe

THEANO_FLAGS='floatX=float32,device=gpu,cuda.root=/share/apps/cuda/6.5.12'
if [ "$PBS_JOBTMP" != "" ]; then
        THEANO_FLAGS="base_compiledir=$PBS_JOBTMP,$THEANO_FLAGS"
fi

export THEANO_FLAGS

cd /scratch/cdg356/spring/scripts

module purge
module load ipdb/0.8
module load pillow/intel/2.7.0
module load pandas/intel/0.16.0
module load cuda/6.5.12
module load cudnn/6.5
module load theano/20150721
module load lasagne/20151007
module load scikit-image/intel/20150129
module load scikit-learn/intel/0.15.2

export LD_LIBRARY_PATH=/share/apps/cudnn/6.5/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/share/apps/cudnn/6.5/lib64:$LIBRARY_PATH
export CPATH=/share/apps/cudnn/6.5/include:$CPATH

#python main.py 10000
#python main.py 10000 use_text
#python main.py 10000 use_images
#python main.py 10000 use_text use_images

#python main.py 50000
#python main.py 50000 use_text
#python main.py 50000 use_images
#python main.py 50000 use_text use_images

#python main.py 100000
#python main.py 100000 use_text
#python main.py 100000 use_images
#python main.py 100000 use_text use_images

#python main.py 200000
python main.py 200000 use_text
python main.py 200000 use_images
python main.py 200000 use_text use_images
