@echo off

set CWD=%cd%
set EXEC_DIR=%CWD%\..\python\tensorflow\

set PYTHON=python
set DATASET=mv_200
set IMG_SZ=224
set NUM_EPOCH=1
set BATCH_SZ=8
set FP16=0
set FMT=NHWC
set model=mobilenet
set MODEL_LOG_DIR=test
set MODEL_PARAM=train,1.0,224,0.8
set LR=0.0001
set L2_REG=0.00005
set DATA_AUG=1
set OPT=adam
set EPOCH=0
set DECAY=0.0
set DECAY_STEP=5

cd %EXEC_DIR%
%PYTHON% deep_nn.py train --model %model% --model-param %MODEL_PARAM% --dataset %DATASET% ^
		--data-format %FMT% --num-epoch %NUM_EPOCH% --batch-size %BATCH_SZ%	--fp16 %FP16% --lr %LR% ^
		--optimizer %OPT% --wd %L2_REG% --log-subdir %MODEL_LOG_DIR% --data-aug %DATA_AUG% ^
		--begin-epoch %EPOCH% --input-size %IMG_SZ% --lr-decay %DECAY% --lr-step %DECAY_STEP%

cd %CWD%