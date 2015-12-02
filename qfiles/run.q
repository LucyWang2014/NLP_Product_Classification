#!/bin/bash

#PBS -l nodes=1:ppn=2:gpus=0
#PBS -l walltime=25:00:00
#PBS -l mem=2GB
#PBS -N img_download
#PBS -j oe

cd /home/cdg356/spring/scripts

module purge
module load pandas/intel/0.16.0
module load pillow/intel/2.7.0

python download_images_to_directory.py
