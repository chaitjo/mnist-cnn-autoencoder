#!/bin/bash
source ~/tensorflow_py27_cpu/bin/activate
cd ~/nn-project-2/CNN/
python cnn_multilayer.py --filters_hidden1=30 --filters_hidden2=30
python cnn_multilayer.py --filters_hidden1=30 --filters_hidden2=65
python cnn_multilayer.py --filters_hidden1=50 --filters_hidden2=30
python cnn_multilayer.py --filters_hidden1=50 --filters_hidden2=50
python cnn_multilayer.py --filters_hidden1=50 --filters_hidden2=65
python cnn_multilayer.py --filters_hidden1=65 --filters_hidden2=30
python cnn_multilayer.py --filters_hidden1=65 --filters_hidden2=50
python cnn_multilayer.py --filters_hidden1=65 --filters_hidden2=65







