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
MODEL_NAME="model_resnet"
DATA_NAME="data_resize"
HR_FLIST="flist/hr.flist"
LR_FLIST="flist/lrX2.flist"
SCALE=2

SCRIPT="train.py"
if [ -n "$SGE_HGR_gpu" ]; then
  source /data/jl_1/shared/ifp/envs/ifp-tf-master/bin/activate
  export LD_LIBRARY_PATH="/home/jl/ifp/yfan/cudnn/lib64":$LD_LIBRARY_PATH
  export CUDA_VISIBLE_DEVICES=`echo $SGE_HGR_gpu | sed 's/GPU//g' | awk -F ' ' '{for(i=1;i<NF;++i)printf "%i,",$i-1; printf "%i",$NF-1}'`
  GPU_NUM=`echo $SGE_HGR_gpu | sed 's/GPU//g' | awk -F ' ' '{printf "%i",NF}'`
  if [ $GPU_NUM -gt 1 ]; then
    SCRIPT="train_multi_async.py --gpu_num=$GPU_NUM --mem_growth=False"
  fi
else
  export CUDA_VISIBLE_DEVICES=0
fi

MODEL_FILE="$TRAIN_DIR/$MODEL_NAME-$DATA_NAME-$EXPR_NAME"
ARGS="--data_name=$DATA_NAME --hr_flist=$HR_FLIST --lr_flist=$LR_FLIST --model_name=$MODEL_NAME --scale=$SCALE"

iter=0
# learning rate 0.001 with adam
for i in `seq 1 3`;
do
    python $SCRIPT $ARGS --model_file_in=$MODEL_FILE-$iter --model_file_out=$MODEL_FILE-$((iter+1)) --learning_rate=0.001
    iter=$((iter+1))
    echo "Iteration $iter Finished"
done

# learning rate 0.0001 with adam
for i in `seq 1 2`;
do
    python $SCRIPT $ARGS --model_file_in=$MODEL_FILE-$iter --model_file_out=$MODEL_FILE-$((iter+1)) --learning_rate=0.0001
    iter=$((iter+1))
    echo "Iteration $iter Finished"
done
