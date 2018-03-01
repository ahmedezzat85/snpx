@echo off

set PYTHON=python
set IMG_SZ=%3
set FMT=NHWC
set model=%1
set MODEL_LOG_DIR=%2
set CHKPT=%4

%PYTHON% deep_nn.py deploy --model-name %model% --data-format %FMT% --log-subdir %MODEL_LOG_DIR% ^
		--input-size %IMG_SZ% --checkpoint %CHKPT%