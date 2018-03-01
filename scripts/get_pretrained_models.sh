#!/bin/sh

Download_DIR=$PWD/../pretrained_models/

mkdir $Download_DIR
sudo chmod 777 $Download_DIR
cd $Download_DIR
mkdir mobilenet_v1
cd mobilenet_v1
wget http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz
tar -xzf mobilenet_v1_1.0_224_2017_06_14.tar.gz

cd $PWD