PYTHON=python
IMG_SZ=224
NUM_EPOCH=1
BATCH_SZ=256
FP16=0
FMT=NHWC
model=mobilenet
MODEL_LOG_DIR=mobilenet-025
MODEL_PARAM=freeze,0.25,224
LR=0.0001
L2_REG=0.0
DATA_AUG=1
OPT=adam
EPOCH=0
DECAY=0.0
DECAY_STEP=5

$PYTHON deep_nn.py train --model-name $model --model-param $MODEL_PARAM --data-format $FMT \
		--num-epoch $NUM_EPOCH --batch-size $BATCH_SZ	--fp16 $FP16 --optimizer $OPT --lr $LR \
		--wd $L2_REG --log-subdir $MODEL_LOG_DIR --data-aug $DATA_AUG --begin-epoch $EPOCH \
		--input-size $IMG_SZ --lr-decay $DECAY --lr-step $DECAY_STEP