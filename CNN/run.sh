#!/bin/bash
source ~/tensorflow_py27_cpu/bin/activate
cd ~/nn-project-2/CNN/
python cnn_multilayer.py --filters_hidden1=25 --filters_hidden2=65
python cnn_multilayer.py --filters_hidden1=35 --filters_hidden2=65








