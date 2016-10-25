#!/bin/bash
source ~/tensorflow_py27_cpu/bin/activate
cd ~/nn-project-2/CNN/
python cnn.py --learning_rate=0.15 --momentum=0.9 --lr_decay=0.95 --batch_size=50
python cnn.py --learning_rate=0.2 --momentum=0.9 --lr_decay=0.95 --batch_size=50
python cnn.py --learning_rate=0.01 --momentum=0.93 --lr_decay=0.95 --batch_size=50
python cnn.py --learning_rate=0.01 --momentum=0.97 --lr_decay=0.95 --batch_size=50
python cnn.py --learning_rate=0.01 --momentum=0.9 --lr_decay=0.93 --batch_size=50
python cnn.py --learning_rate=0.01 --momentum=0.9 --lr_decay=0.97 --batch_size=50
