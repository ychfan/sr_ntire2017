#!/bin/bash

# Specify the name for your job name, this is the job name by which grid engine will 
# refer to your job, this could be different from name of your executable or name of your script file
#$ -N sr
#
# Join the standard output & error files into one file (y)[yes] or write to separate files (n)[no]
# The default is n [no]
#$ -j y
#
# Use the directory from where the job is submitted
#$ -cwd
#
# The output path and file name if different from job name
# #$ -o coutput
#
# Specify the number of GPU for your job
#$ -l gpu=1
#
# Specify the hostname of the machine to run
#$ -l h="*1|*2|*3"

set -x

EXPR_NAME="try"
TRAIN_DIR="tmp"
MODEL_NAME="model_res_pre_act"
DATA_NAME="data_naive_bc"
HR_FLIST="flist/hr.flist"
LR_FLIST="flist/lrX2.flist"

source /data/jl_1/shared/ifp/envs/ifp-tf-master/bin/activate
export CUDA_VISIBLE_DEVICES=`echo $SGE_HGR_gpu | sed 's/GPU//g' | awk -F ' ' '{for(i=1;i<NF;++i)printf "%i,",$i-1; printf "%i",$NF-1}'`
#export CUDA_VISIBLE_DEVICES=0

MODEL_FILE="$TRAIN_DIR/$MODEL_NAME-$DATA_NAME-$EXPR_NAME"
ARGS="--data_name=$DATA_NAME --hr_flist=$HR_FLIST --lr_flist=$LR_FLIST --model_name=$MODEL_NAME --model_file=$MODEL_FILE"
# learning rate 0.001 with adam
for i in `seq 1 10`;
do
    python train.py $ARGS --learning_rate=0.001
done

# learning rate 0.0001 with adam
for i in `seq 1 2`;
do
    python train.py $ARGS --learning_rate=0.0001
done
