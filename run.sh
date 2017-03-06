#!bin/bash

EXPR_NAME="try"
TRAIN_DIR="tmp/"
MODEL_NAME="model_res_pre_act"


export CUDA_VISIBLE_DEVICES=0

MODEL_FILE="$TRAIN_DIR/$MODEL_NAME_$EXPR_NAME"
# learning rate 0.01 with adam
python train.py --model_name=$MODEL_NAME --model_file=$MODEL_FILE --learning_rate=0.01

# learning rate 0.001 with adam
for i in `seq 1 10`;
do
    python train.py --model_name=$MODEL_NAME --model_file=$MODEL_FILE --learning_rate=0.001
done

# learning rate 0.0001 with adam
for i in `seq 1 2`;
do
    python train.py --model_name=$MODEL_NAME --model_file=$MODEL_FILE --learning_rate=0.0001
done