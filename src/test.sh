#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=1
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

#DATA_DIR=/opt/jiaguo/faces_vgg_112x112
#DATA_DIR=/data/face/DeepGlint/images/glint_112x112
#DATA_DIR=/data/face/MS_Celeb_1M/images/ArcFace/faces_ms1m_112x112
DATA_DIR=/data/face/DeepGlint/images/glint_celebrity_112x112/glint_cn,/data/face/MS_Celeb_1M/images/ArcFace/faces_emore
#DATA_DIR=/data/face/SensingtechFaces/chxd/images/mxnet/yc1,/data/face/MS_Celeb_1M/images/ArcFace/faces_emore

NETWORK=y72
JOB=lala
MODELDIR="../models/emore_glint-model-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
cp ./train.sh $MODELDIR/
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"
LRSTEPS='100000,140000,160000'
#python -u train_softmax.py --lr 0.1 --fc7-lr-mult 1 --fc7-wd-mult 10 --lr-steps "$LRSTEPS" --data-dir $DATA_DIR --network "$NETWORK" --width-mult 1 --loss-type 0 --margin-m 0.5 --prefix "$PREFIX" --per-batch-size 64 --verbose 20000 --ckpt 2 --max-steps  200001 --version-output "GDC" # > "$LOGFILE" 2>&1 &
#python -u train_softmax.py --lr 0.01 --fc7-lr-mult 1 --fc7-wd-mult 10 --lr-steps "$LRSTEPS" --data-dir $DATA_DIR --network "$NETWORK" --width-mult 1 --loss-type 4 --margin-m 0.35 --prefix "$PREFIX" --per-batch-size 64 --verbose 20000 --ckpt 2 --max-steps  200001 --pretrained "$PREFIX,10" --version-output "GDC" # > "$LOGFILE" 2>&1 &
#python -u train_softmax.py --lr 0.01 --fc7-lr-mult 1 --fc7-wd-mult 10 --lr-steps "$LRSTEPS" --data-dir $DATA_DIR --network "$NETWORK" --width-mult 1 --loss-type 4 --margin-m 0.5 --prefix "$PREFIX" --per-batch-size 64 --verbose 20000 --ckpt 2 --max-steps  200001 --pretrained "$PREFIX,10" --version-output "GDC" # > "$LOGFILE" 2>&1 &
LRSTEPS='400000,600000'
python -u train_softmax.py --lr 0.1 --fc7-lr-mult 1 --fc7-wd-mult 10 --lr-steps "$LRSTEPS" --data-dir $DATA_DIR --network "$NETWORK" --width-mult 1 --loss-type 6 --margin-m 0.5 --prefix "$PREFIX" --per-batch-size 64 --verbose 20000 --ckpt 2 --max-steps  800001 --version-output "GDC" --margin-policy 'linear' # > "$LOGFILE" 2>&1 &

#python -u train_softmax.py --lr-steps "$LRSTEPS" --data-dir $DATA_DIR --network "$NETWORK" --prefix "$PREFIX" --width-mult 1 \
#           --fc7-lr-mult 1 --fc7-wd-mult 10 --loss-type 0 --margin-m 0.5 --lr 0.1 \
#           --per-batch-size 32 --verbose 20000 --ckpt 2 --max-steps  200001 # > "$LOGFILE" 2>&1 &
#python -u train_softmax.py --lr-steps "$LRSTEPS" --data-dir $DATA_DIR --network "$NETWORK" --prefix "$PREFIX" --width-mult 1 \
#           --fc7-lr-mult 1 --fc7-wd-mult 10 --loss-type 4 --margin-m 0.35 --lr 0.01 \
#           --per-batch-size 32 --verbose 20000 --ckpt 2 --max-steps  200001 --pretrained "$PREFIX,10" # > "$LOGFILE" 2>&1 &
#LRSTEPS='400000,600000'
#python -u train_softmax.py --lr-steps "$LRSTEPS" --data-dir $DATA_DIR --network "$NETWORK" --prefix "$PREFIX" --width-mult 1 \
#           --fc7-lr-mult 1 --fc7-wd-mult 10 --loss-type 4 --margin-m 0.5 --lr 0.01 \
#           --per-batch-size 32 --verbose 40000 --ckpt 2 --max-steps  800001 --pretrained "$PREFIX,10" # > "$LOGFILE" 2>&1 &
